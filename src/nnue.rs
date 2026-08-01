//! NNUE evaluation (build plan, Phase 4).
//!
//! Topology: `768 -> 256` per perspective (weights shared) -> concatenate to 512
//! -> `32` -> `32` -> `1`. ClippedReLU throughout. ~200k parameters.
//!
//! **No incremental accumulator updates.** The usual NNUE efficiency argument is
//! that a chess move changes one or two features, so you patch the accumulator.
//! Here a single blast changes up to ten features at once, and captures are
//! exactly the nodes quiescence spends its time in. A full refresh is ~32 columns
//! of 256 floats — cheap enough to do at every node. Measure before optimising.
//!
//! **No ply-count input.** The ply cap is handled by `Position::result`
//! adjudicating at the horizon, so the network never needs to know the ply.
//!
//! Precision is `f32`. Quantise to int16/int8 only if profiling says evaluation
//! rather than movegen is the bottleneck.

use crate::eval::MATE_THRESHOLD;
use crate::position::Position;
use crate::types::*;
use std::io::{Read, Write};
use std::path::Path;

pub const NUM_FEATURES: usize = 768; // 12 piece types x 64 squares
/// Largest hidden layer the stack buffers are sized for.
pub const MAX_HL: usize = 256;
/// Hidden layer per perspective for a freshly built network. The actual width is
/// read from the file header, so one binary can load and compare networks of
/// different sizes — which is the only way to find out how wide this variant
/// actually needs.
pub const DEFAULT_HL: usize = 256;
pub const L1: usize = 32;
pub const L2: usize = 32;

const MAGIC: u32 = 0x4e_4e_53_42; // "BSNN" little-endian
const VERSION: u32 = 1;

/// Feature index for one piece, from `perspective`'s point of view.
///
/// The relative colour bit is all that is needed to canonicalise — there is no
/// board flip, because no piece in this variant has a forward direction.
#[inline(always)]
pub fn feature_index(perspective: Color, piece_color: Color, pt: PieceType, square: usize) -> usize {
    let relative = if piece_color == perspective { 0 } else { 1 };
    (relative * NUM_PIECE_TYPES + pt.index()) * 64 + square
}

#[derive(Clone)]
pub struct Network {
    /// Hidden layer width per perspective, from the file header.
    pub hl: usize,
    /// Feature transformer, row-major by feature so one feature is a contiguous
    /// `hl`-float span.
    pub ft_weight: Vec<f32>, // NUM_FEATURES * hl
    pub ft_bias: Vec<f32>,   // hl
    pub w1: Vec<f32>,        // L1 * (2 * hl)
    pub b1: Vec<f32>,        // L1
    pub w2: Vec<f32>,        // L2 * L1
    pub b2: Vec<f32>,        // L2
    pub w3: Vec<f32>,        // 1 * L2
    pub b3: Vec<f32>,        // 1
}

#[inline(always)]
fn crelu(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Vector width to encourage LLVM towards. 8 f32 lanes is one AVX2 register.
const LANES: usize = 8;

/// Dot product, summed in `LANES` independent partial sums.
///
/// The obvious `for i { sum += a[i] * b[i] }` does **not** vectorise: float
/// addition is not associative, so LLVM may not legally reorder a single-variable
/// reduction, and it silently emits scalar code. Splitting into independent
/// partial sums makes the reassociation explicit and lets it use FMA.
///
/// This is the whole reason NNUE evaluation was running at scalar speed even with
/// `target-cpu=native` — the vector units were available and simply unused.
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sums = [0f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        for j in 0..LANES {
            sums[j] = x[j].mul_add(y[j], sums[j]);
        }
    }
    let mut total = 0.0;
    for s in sums {
        total += s;
    }
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        total += x * y;
    }
    total
}

/// `dst += src`, element-wise. Iterator zip keeps the bounds checks out and this
/// one does vectorise on its own, once the slice lengths are known to match.
#[inline(always)]
fn add_assign(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

impl Network {
    pub fn zeroed() -> Network {
        Network::zeroed_with(DEFAULT_HL)
    }

    pub fn zeroed_with(hl: usize) -> Network {
        assert!(hl <= MAX_HL && hl % 8 == 0, "hidden layer must be a multiple of 8, at most {MAX_HL}");
        Network {
            hl,
            ft_weight: vec![0.0; NUM_FEATURES * hl],
            ft_bias: vec![0.0; hl],
            w1: vec![0.0; L1 * 2 * hl],
            b1: vec![0.0; L1],
            w2: vec![0.0; L2 * L1],
            b2: vec![0.0; L2],
            w3: vec![0.0; L2],
            b3: vec![0.0; 1],
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.ft_weight.len()
            + self.ft_bias.len()
            + self.w1.len()
            + self.b1.len()
            + self.w2.len()
            + self.b2.len()
            + self.w3.len()
            + self.b3.len()
    }

    /// Full accumulator refresh for both perspectives. Only the first `self.hl`
    /// entries of each half are written.
    pub fn accumulate(&self, pos: &Position, acc: &mut [[f32; MAX_HL]; 2]) {
        for half in acc.iter_mut() {
            half[..self.hl].copy_from_slice(&self.ft_bias);
        }
        let (white, black) = acc.split_at_mut(1);
        for p in 0..NUM_PIECES {
            let board = pos.pieces[p];
            if board == 0 {
                continue;
            }
            let pc = piece_color(p);
            let pt = piece_type_of(p);
            // The two perspectives differ only in the relative-colour bit, so the
            // feature indices are a fixed distance apart — no need to recompute.
            let fw = feature_index(Color::White, pc, pt, 0);
            let fb = feature_index(Color::Black, pc, pt, 0);
            for s in bits(board) {
                add_assign(&mut white[0][..self.hl], self.column(fw + s));
                add_assign(&mut black[0][..self.hl], self.column(fb + s));
            }
        }
    }

    #[inline(always)]
    fn column(&self, feature: usize) -> &[f32] {
        &self.ft_weight[feature * self.hl..(feature + 1) * self.hl]
    }

    /// Centipawns from the side to move's point of view.
    pub fn evaluate(&self, pos: &Position) -> i32 {
        let mut acc = [[0.0f32; MAX_HL]; 2];
        self.accumulate(pos, &mut acc);

        // Side to move's perspective goes first.
        let (us, them) = match pos.side {
            Color::White => (0, 1),
            Color::Black => (1, 0),
        };
        let hl = self.hl;
        let mut input = [0.0f32; 2 * MAX_HL];
        for i in 0..hl {
            input[i] = crelu(acc[us][i]);
            input[hl + i] = crelu(acc[them][i]);
        }
        let input = &input[..2 * hl];

        let mut h1 = [0.0f32; L1];
        for (o, h) in h1.iter_mut().enumerate() {
            let row = &self.w1[o * 2 * hl..(o + 1) * 2 * hl];
            *h = crelu(self.b1[o] + dot(row, input));
        }

        let mut h2 = [0.0f32; L2];
        for (o, h) in h2.iter_mut().enumerate() {
            let row = &self.w2[o * L1..(o + 1) * L1];
            *h = crelu(self.b2[o] + dot(row, &h1));
        }

        let out = self.b3[0] + dot(&self.w3, &h2);

        // The network is trained directly in centipawns (see tools/train.py).
        (out as i32).clamp(-MATE_THRESHOLD + 1, MATE_THRESHOLD - 1)
    }

    // -----------------------------------------------------------------------
    // Serialisation
    // -----------------------------------------------------------------------

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Network> {
        let mut buf = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut buf)?;
        Network::from_bytes(&buf)
    }

    pub fn from_bytes(buf: &[u8]) -> std::io::Result<Network> {
        use std::io::{Error, ErrorKind};
        let bad = |m: &str| Error::new(ErrorKind::InvalidData, m.to_string());
        if buf.len() < 24 {
            return Err(bad("network file too short"));
        }
        let u32_at = |o: usize| {
            u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]])
        };
        if u32_at(0) != MAGIC {
            return Err(bad("not a bschess network file"));
        }
        if u32_at(4) != VERSION {
            return Err(bad("unsupported network version"));
        }
        let (nf, hl, l1, l2) = (
            u32_at(8) as usize,
            u32_at(12) as usize,
            u32_at(16) as usize,
            u32_at(20) as usize,
        );
        // The hidden layer width is whatever the file says, so networks of
        // different sizes can be loaded and played against each other.
        if (nf, l1, l2) != (NUM_FEATURES, L1, L2) {
            return Err(bad(&format!(
                "network shape {nf}x{hl}x{l1}x{l2} does not match the compiled \
                 {NUM_FEATURES}x*x{L1}x{L2}"
            )));
        }
        if hl == 0 || hl > MAX_HL || hl % 8 != 0 {
            return Err(bad(&format!(
                "hidden layer {hl} must be a non-zero multiple of 8, at most {MAX_HL}"
            )));
        }
        let mut net = Network::zeroed_with(hl);
        let mut off = 24;
        let read = |dst: &mut Vec<f32>, off: &mut usize| -> std::io::Result<()> {
            let n = dst.len();
            if buf.len() < *off + n * 4 {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "network file truncated".to_string(),
                ));
            }
            for (i, slot) in dst.iter_mut().enumerate() {
                let o = *off + i * 4;
                *slot = f32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
            }
            *off += n * 4;
            Ok(())
        };
        read(&mut net.ft_weight, &mut off)?;
        read(&mut net.ft_bias, &mut off)?;
        read(&mut net.w1, &mut off)?;
        read(&mut net.b1, &mut off)?;
        read(&mut net.w2, &mut off)?;
        read(&mut net.b2, &mut off)?;
        read(&mut net.w3, &mut off)?;
        read(&mut net.b3, &mut off)?;
        Ok(net)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(&MAGIC.to_le_bytes())?;
        f.write_all(&VERSION.to_le_bytes())?;
        for v in [NUM_FEATURES, self.hl, L1, L2] {
            f.write_all(&(v as u32).to_le_bytes())?;
        }
        for arr in [
            &self.ft_weight,
            &self.ft_bias,
            &self.w1,
            &self.b1,
            &self.w2,
            &self.b2,
            &self.w3,
            &self.b3,
        ] {
            for x in arr.iter() {
                f.write_all(&x.to_le_bytes())?;
            }
        }
        f.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_count_is_about_200k() {
        let n = Network::zeroed();
        assert_eq!(n.parameter_count(), 768 * 256 + 256 + 32 * 512 + 32 + 32 * 32 + 32 + 32 + 1);
        assert!((190_000..230_000).contains(&n.parameter_count()));
    }

    #[test]
    fn feature_indices_are_distinct_and_in_range() {
        let mut seen = std::collections::HashSet::new();
        for c in [Color::White, Color::Black] {
            for pt in ALL_PIECE_TYPES {
                for s in 0..64 {
                    let f = feature_index(Color::White, c, pt, s);
                    assert!(f < NUM_FEATURES);
                    assert!(seen.insert(f), "duplicate feature index {f}");
                }
            }
        }
        assert_eq!(seen.len(), NUM_FEATURES);
    }

    #[test]
    fn perspectives_mirror_each_other() {
        // The same piece is "theirs" from one perspective and "ours" from the
        // other; the relative colour bit is the whole difference.
        const OURS: usize = 0;
        const THEIRS: usize = 1;
        let rook = PieceType::Rook.index();
        let square = 12;

        let f_white = feature_index(Color::White, Color::Black, PieceType::Rook, square);
        let f_black = feature_index(Color::Black, Color::Black, PieceType::Rook, square);

        assert_eq!(f_white, (THEIRS * NUM_PIECE_TYPES + rook) * 64 + square);
        assert_eq!(f_black, (OURS * NUM_PIECE_TYPES + rook) * 64 + square);
        assert_eq!(f_white - f_black, NUM_PIECE_TYPES * 64);
    }

    #[test]
    fn save_load_roundtrip() {
        let mut net = Network::zeroed();
        for (i, w) in net.ft_weight.iter_mut().enumerate() {
            *w = (i % 97) as f32 / 97.0;
        }
        net.b3[0] = 1.5;
        let dir = std::env::temp_dir().join("bschess_net_test.bin");
        net.save(&dir).unwrap();
        let back = Network::load(&dir).unwrap();
        assert_eq!(back.ft_weight, net.ft_weight);
        assert_eq!(back.b3, net.b3);
        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn zero_network_evaluates_to_zero() {
        let net = Network::zeroed();
        assert_eq!(net.evaluate(&Position::startpos()), 0);
    }
}
