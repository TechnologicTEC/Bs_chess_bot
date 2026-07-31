//! Precomputed lookup tables (build plan, Phase 1).
//!
//! | Table                  | Size          | Purpose                              |
//! |------------------------|---------------|--------------------------------------|
//! | 3x3 explosion masks    | 64 x u64      | rook/bishop detonation regions       |
//! | knight path masks      | 64 x 8 x u64  | both intermediates + destination     |
//! | ray tables             | 8 x 64 x u64  | classical sliding attacks w/ blockers|
//! | between masks          | 64 x 64 x u64 | queen sweep + ray-block detection    |
//! | zobrist keys           | 12 x 64 + 1   | transposition table                  |
//!
//! Sliding attacks use classical ray-scan rather than magics. The variant's queen
//! needs a bespoke enumeration anyway (spec §2.4), so magics would only accelerate
//! rook/bishop generation, which is not the bottleneck here — pawn teleport
//! enumeration is. Ray-scan is branch-light and obviously correct.

use crate::types::*;
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Directions
// ---------------------------------------------------------------------------

pub const DIR_N: usize = 0;
pub const DIR_NE: usize = 1;
pub const DIR_E: usize = 2;
pub const DIR_SE: usize = 3;
pub const DIR_S: usize = 4;
pub const DIR_SW: usize = 5;
pub const DIR_W: usize = 6;
pub const DIR_NW: usize = 7;

/// `(d_rank, d_file)` for each of the 8 ray directions, indexed by `DIR_*`.
pub const DELTAS: [(i32, i32); 8] = [
    (1, 0),   // N
    (1, 1),   // NE
    (0, 1),   // E
    (-1, 1),  // SE
    (-1, 0),  // S
    (-1, -1), // SW
    (0, -1),  // W
    (1, -1),  // NW
];

/// Directions whose square index increases (so the *first* blocker is the LSB).
const POSITIVE_DIR: [bool; 8] = [
    true,  // N  +8
    true,  // NE +9
    true,  // E  +1
    false, // SE -7
    false, // S  -8
    false, // SW -9
    false, // W  -1
    true,  // NW +7
];

pub const ROOK_DIRS: [usize; 4] = [DIR_N, DIR_E, DIR_S, DIR_W];
pub const BISHOP_DIRS: [usize; 4] = [DIR_NE, DIR_SE, DIR_SW, DIR_NW];
pub const QUEEN_DIRS: [usize; 8] = [
    DIR_N, DIR_NE, DIR_E, DIR_SE, DIR_S, DIR_SW, DIR_W, DIR_NW,
];

// ---------------------------------------------------------------------------
// Knight path geometry — spec §2.2
// ---------------------------------------------------------------------------

/// `(dest_dr, dest_dc, i1_dr, i1_dc, i2_dr, i2_dc)`, transcribed directly from
/// the spec's path table. Two steps along the major axis, then one perpendicular.
pub const KNIGHT_PATHS: [(i32, i32, i32, i32, i32, i32); 8] = [
    (2, 1, 1, 0, 2, 0),
    (2, -1, 1, 0, 2, 0),
    (-2, 1, -1, 0, -2, 0),
    (-2, -1, -1, 0, -2, 0),
    (1, 2, 0, 1, 0, 2),
    (-1, 2, 0, 1, 0, 2),
    (1, -2, 0, -1, 0, -2),
    (-1, -2, 0, -1, 0, -2),
];

// ---------------------------------------------------------------------------
// Table storage
// ---------------------------------------------------------------------------

pub struct Tables {
    /// 3x3 region centred on the square, clipped at the edges. Includes the centre.
    pub blast: [Bitboard; 64],
    /// The up-to-8 neighbours of a square, excluding the square itself.
    pub ring: [Bitboard; 64],
    /// All squares in direction `d` from `s`, exclusive of `s`.
    pub rays: [[Bitboard; 64]; 8],
    /// Squares strictly between two aligned squares; 0 if not aligned.
    pub between: [[Bitboard; 64]; 64],
    /// Direction index from `a` to `b`, or 8 if not aligned.
    pub dir_between: [[u8; 64]; 64],
    /// Per square: for each of the 8 knight move kinds, `(dest, path_mask)`.
    /// `dest == 64` marks an off-board (illegal) knight move.
    pub knight_dest: [[u8; 8]; 64],
    pub knight_path: [[Bitboard; 8]; 64],
    /// `knight_path_lookup[from][to]` is the 3-square path mask, or 0 when `to`
    /// is not a knight destination from `from`. 32 KiB, keeps `make_move` branchless.
    pub knight_path_lookup: Box<[[Bitboard; 64]; 64]>,
    /// Union of all knight destinations from a square (for quick "is a knight
    /// move to X legal" checks).
    pub knight_targets: [Bitboard; 64],
    /// Union of all path squares reachable by any knight move from a square.
    pub knight_span: [Bitboard; 64],
    /// Squares along direction `d` from `s`, in outward order. `ray_len` gives the
    /// count. Used by the queen's ordered walk (spec §2.4), which cannot be
    /// expressed as a plain attack set.
    pub ray_sq: [[[u8; 7]; 64]; 8],
    pub ray_len: [[u8; 64]; 8],
    /// Pseudo attacks on an empty board (used only as a fast reject).
    pub rook_pseudo: [Bitboard; 64],
    pub bishop_pseudo: [Bitboard; 64],
    /// Zobrist keys.
    pub zobrist_piece: [[u64; 64]; NUM_PIECES],
    pub zobrist_side: u64,
}

fn on_board(r: i32, f: i32) -> bool {
    (0..8).contains(&r) && (0..8).contains(&f)
}

fn build() -> Tables {
    let mut t = Tables {
        blast: [0; 64],
        ring: [0; 64],
        rays: [[0; 64]; 8],
        between: [[0; 64]; 64],
        dir_between: [[8; 64]; 64],
        knight_dest: [[64; 8]; 64],
        knight_path: [[0; 8]; 64],
        knight_path_lookup: Box::new([[0; 64]; 64]),
        knight_targets: [0; 64],
        knight_span: [0; 64],
        ray_sq: [[[0; 7]; 64]; 8],
        ray_len: [[0; 64]; 8],
        rook_pseudo: [0; 64],
        bishop_pseudo: [0; 64],
        zobrist_piece: [[0; 64]; NUM_PIECES],
        zobrist_side: 0,
    };

    // --- 3x3 blast masks and king rings ------------------------------------
    for s in 0..64 {
        let (r, f) = (rank_of(s) as i32, file_of(s) as i32);
        let mut blast = 0u64;
        for dr in -1..=1 {
            for df in -1..=1 {
                if on_board(r + dr, f + df) {
                    blast |= bb(sq((r + dr) as usize, (f + df) as usize));
                }
            }
        }
        t.blast[s] = blast;
        t.ring[s] = blast & !bb(s);
    }

    // --- rays ---------------------------------------------------------------
    for s in 0..64 {
        let (r, f) = (rank_of(s) as i32, file_of(s) as i32);
        for (d, &(dr, df)) in DELTAS.iter().enumerate() {
            let (mut rr, mut ff) = (r + dr, f + df);
            let mut mask = 0u64;
            let mut n = 0usize;
            while on_board(rr, ff) {
                let x = sq(rr as usize, ff as usize);
                mask |= bb(x);
                t.ray_sq[d][s][n] = x as u8;
                n += 1;
                rr += dr;
                ff += df;
            }
            t.rays[d][s] = mask;
            t.ray_len[d][s] = n as u8;
        }
        for &d in ROOK_DIRS.iter() {
            t.rook_pseudo[s] |= t.rays[d][s];
        }
        for &d in BISHOP_DIRS.iter() {
            t.bishop_pseudo[s] |= t.rays[d][s];
        }
    }

    // --- between / direction ------------------------------------------------
    for a in 0..64 {
        let (r, f) = (rank_of(a) as i32, file_of(a) as i32);
        for (d, &(dr, df)) in DELTAS.iter().enumerate() {
            let (mut rr, mut ff) = (r + dr, f + df);
            let mut acc = 0u64;
            while on_board(rr, ff) {
                let b = sq(rr as usize, ff as usize);
                t.between[a][b] = acc;
                t.dir_between[a][b] = d as u8;
                acc |= bb(b);
                rr += dr;
                ff += df;
            }
        }
    }

    // --- knight paths -------------------------------------------------------
    for s in 0..64 {
        let (r, f) = (rank_of(s) as i32, file_of(s) as i32);
        for (k, &(ddr, ddf, i1r, i1f, i2r, i2f)) in KNIGHT_PATHS.iter().enumerate() {
            let (dr, df) = (r + ddr, f + ddf);
            if !on_board(dr, df) {
                continue;
            }
            // The spec's geometry guarantees both intermediates are on-board
            // whenever the destination is; assert it rather than assume it.
            debug_assert!(on_board(r + i1r, f + i1f) && on_board(r + i2r, f + i2f));
            let dest = sq(dr as usize, df as usize);
            let path = bb(dest)
                | bb(sq((r + i1r) as usize, (f + i1f) as usize))
                | bb(sq((r + i2r) as usize, (f + i2f) as usize));
            t.knight_dest[s][k] = dest as u8;
            t.knight_path[s][k] = path;
            t.knight_path_lookup[s][dest] = path;
            t.knight_targets[s] |= bb(dest);
            t.knight_span[s] |= path;
        }
    }

    // --- zobrist ------------------------------------------------------------
    let mut rng = SplitMix64::new(0x9E3779B97F4A7C15);
    for p in 0..NUM_PIECES {
        for s in 0..64 {
            t.zobrist_piece[p][s] = rng.next();
        }
    }
    t.zobrist_side = rng.next();

    t
}

pub static TABLES: LazyLock<Tables> = LazyLock::new(build);

/// Force table construction (call once at startup so the first search does not
/// pay for it inside its clock).
pub fn init() {
    LazyLock::force(&TABLES);
}

// ---------------------------------------------------------------------------
// Sliding attacks
// ---------------------------------------------------------------------------

/// Squares reachable along direction `d` from `s`, stopping *on* the first
/// occupied square (which is included — landing there is the detonating move).
#[inline(always)]
pub fn ray_attacks(d: usize, s: usize, occ: Bitboard) -> Bitboard {
    let t = &*TABLES;
    let ray = t.rays[d][s];
    let blockers = ray & occ;
    if blockers == 0 {
        return ray;
    }
    let first = if POSITIVE_DIR[d] {
        blockers.trailing_zeros() as usize
    } else {
        63 - blockers.leading_zeros() as usize
    };
    ray ^ t.rays[d][first]
}

#[inline(always)]
pub fn rook_attacks(s: usize, occ: Bitboard) -> Bitboard {
    ROOK_DIRS.iter().fold(0, |a, &d| a | ray_attacks(d, s, occ))
}

#[inline(always)]
pub fn bishop_attacks(s: usize, occ: Bitboard) -> Bitboard {
    BISHOP_DIRS
        .iter()
        .fold(0, |a, &d| a | ray_attacks(d, s, occ))
}

/// Rook/bishop reach for the given piece type. Queens do *not* use this — see
/// `movegen::queen_targets`.
#[inline(always)]
pub fn slider_attacks(pt: PieceType, s: usize, occ: Bitboard) -> Bitboard {
    match pt {
        PieceType::Rook => rook_attacks(s, occ),
        PieceType::Bishop => bishop_attacks(s, occ),
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Deterministic RNG (also used by self-play opening randomisation)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }
    #[inline(always)]
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    #[inline(always)]
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
    #[inline(always)]
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blast_corner_is_2x2() {
        let t = &*TABLES;
        assert_eq!(t.blast[parse_square("a1").unwrap()].count_ones(), 4);
        assert_eq!(t.blast[parse_square("h8").unwrap()].count_ones(), 4);
        assert_eq!(t.blast[parse_square("a4").unwrap()].count_ones(), 6);
        assert_eq!(t.blast[parse_square("d4").unwrap()].count_ones(), 9);
    }

    #[test]
    fn knight_paths_match_spec_table() {
        let t = &*TABLES;
        // From d4 the (+2,+1) move goes d4 -> d5 -> d6 -> e6.
        let d4 = parse_square("d4").unwrap();
        let k = 0; // (2, 1)
        assert_eq!(t.knight_dest[d4][k] as usize, parse_square("e6").unwrap());
        let expect = bb(parse_square("d5").unwrap())
            | bb(parse_square("d6").unwrap())
            | bb(parse_square("e6").unwrap());
        assert_eq!(t.knight_path[d4][k], expect);
        // Every legal knight move has a 3-square path.
        for s in 0..64 {
            for kk in 0..8 {
                if t.knight_dest[s][kk] != 64 {
                    assert_eq!(t.knight_path[s][kk].count_ones(), 3);
                }
            }
        }
    }

    #[test]
    fn ray_stops_on_first_blocker_inclusive() {
        let a1 = parse_square("a1").unwrap();
        let a4 = parse_square("a4").unwrap();
        let occ = bb(a4);
        let att = ray_attacks(DIR_N, a1, occ);
        let expect = bb(parse_square("a2").unwrap()) | bb(parse_square("a3").unwrap()) | bb(a4);
        assert_eq!(att, expect);
    }

    #[test]
    fn between_is_exclusive() {
        let t = &*TABLES;
        let a1 = parse_square("a1").unwrap();
        let a4 = parse_square("a4").unwrap();
        assert_eq!(
            t.between[a1][a4],
            bb(parse_square("a2").unwrap()) | bb(parse_square("a3").unwrap())
        );
        assert_eq!(t.between[a1][parse_square("a2").unwrap()], 0);
        assert_eq!(t.dir_between[a1][parse_square("b3").unwrap()], 8);
    }
}
