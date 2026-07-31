//! Move generation — spec §2.
//!
//! Everything generated here is legal. There is no check, no pin detection and no
//! king-safety filtering (spec §1.1), and a side to move always has at least one
//! move (spec §1.4) — an empty move list is a bug, not a stalemate.

use crate::position::Position;
use crate::tables::*;
use crate::types::*;

/// Upper bound on the move count. The theoretical maximum is 8 pawns x 47 empty
/// squares + ~105 piece moves = 481, reached with a full 16-piece side against a
/// bare king. No promotion exists, so a side can never hold more than 8 pawns.
pub const MAX_MOVES: usize = 512;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GenMode {
    /// Every legal move, including all `pawns x empty_squares` teleports.
    /// Correct but unsearchable; this is what perft uses.
    All,
    /// Piece moves plus the restricted teleport set (see `teleport_targets`).
    Search,
    /// Captures only — quiescence. Pawns never capture, so they contribute none.
    Captures,
}

#[derive(Clone)]
pub struct MoveList {
    pub moves: [Move; MAX_MOVES],
    pub scores: [i32; MAX_MOVES],
    pub len: usize,
}

impl Default for MoveList {
    fn default() -> Self {
        MoveList::new()
    }
}

impl MoveList {
    pub fn new() -> MoveList {
        MoveList {
            moves: [Move::NONE; MAX_MOVES],
            scores: [0; MAX_MOVES],
            len: 0,
        }
    }
    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < MAX_MOVES, "move list overflow");
        if self.len < MAX_MOVES {
            self.moves[self.len] = m;
            self.len += 1;
        }
    }
    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }
    pub fn contains(&self, m: Move) -> bool {
        self.as_slice().contains(&m)
    }
    pub fn to_vec(&self) -> Vec<Move> {
        self.as_slice().to_vec()
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn generate(pos: &Position, mode: GenMode, list: &mut MoveList) {
    list.clear();
    let us = pos.side;
    let occ = pos.occ;
    let t = &*TABLES;
    let caps_only = mode == GenMode::Captures;

    // --- King (§2.5): one square, any direction, may take friendlies ---------
    for from in bits(pos.bb_of(us, PieceType::King)) {
        let mut targets = t.ring[from];
        if caps_only {
            targets &= occ;
        }
        for to in bits(targets) {
            list.push(Move::new(from, to));
        }
    }

    // --- Knight (§2.2): unblockable sweep over a 3-square path ---------------
    for from in bits(pos.bb_of(us, PieceType::Knight)) {
        for k in 0..8 {
            let dest = t.knight_dest[from][k];
            if dest == 64 {
                continue;
            }
            if caps_only && t.knight_path[from][k] & occ == 0 {
                continue;
            }
            list.push(Move::new(from, dest as usize));
        }
    }

    // --- Rook / Bishop (§2.3): ray move, detonate on the first blocker -------
    for (pt, dirs) in [
        (PieceType::Rook, &ROOK_DIRS[..]),
        (PieceType::Bishop, &BISHOP_DIRS[..]),
    ] {
        for from in bits(pos.bb_of(us, pt)) {
            let mut targets = dirs
                .iter()
                .fold(0u64, |a, &d| a | ray_attacks(d, from, occ));
            if caps_only {
                // Only landing on an occupied square detonates; quiet ray moves
                // change no material.
                targets &= occ;
            }
            for to in bits(targets) {
                list.push(Move::new(from, to));
            }
        }
    }

    // --- Queen (§2.4): forced sweeper, hard stop after the 2nd capture -------
    // Transcribed directly from the spec's enumeration pseudocode.
    for from in bits(pos.bb_of(us, PieceType::Queen)) {
        for &d in QUEEN_DIRS.iter() {
            let n = t.ray_len[d][from] as usize;
            let mut captures = 0;
            for i in 0..n {
                let to = t.ray_sq[d][from][i] as usize;
                let occupied = occ & bb(to) != 0;
                if !caps_only || captures > 0 || occupied {
                    list.push(Move::new(from, to));
                }
                if occupied {
                    captures += 1;
                    if captures == 2 {
                        break;
                    }
                }
            }
        }
    }

    // --- Pawn (§2.1): teleport to any empty square, never captures ----------
    if !caps_only {
        let pawns = pos.bb_of(us, PieceType::Pawn);
        if pawns != 0 {
            let targets = match mode {
                GenMode::All => pos.empty_squares(),
                _ => teleport_targets(pos, us),
            };
            for from in bits(pawns) {
                for to in bits(targets) {
                    // A pawn occupies its own square, so `targets` (a subset of
                    // the empty squares) can never contain `from`. No null move.
                    list.push(Move::new(from, to));
                }
            }
        }
    }

    debug_assert!(
        mode == GenMode::Captures || !list.is_empty(),
        "empty move list — spec §1.4 says this cannot happen"
    );
}

pub fn generate_vec(pos: &Position, mode: GenMode) -> Vec<Move> {
    let mut l = MoveList::new();
    generate(pos, mode, &mut l);
    l.to_vec()
}

// ---------------------------------------------------------------------------
// Restricted teleport generation (build plan, Phase 1)
// ---------------------------------------------------------------------------

/// `pawns x empty_squares` is ~250 in the opening, which makes alpha-beta
/// hopeless. Restrict destinations to the ones that can matter:
///
/// 1. adjacent to either king — creating or denying a detonator,
/// 2. on a ray between an enemy slider and our king — blocking a mate threat,
/// 3. adjacent to a friendly rook/bishop's reach where detonating would pay,
/// 4. a small spread sample of neutral parking squares, so a pawn sitting next to
///    our own king (a liability, spec §5.1) can always be evacuated.
///
/// This cuts ~250 targets to roughly 15-25. `GenMode::All` keeps the full set for
/// perft.
pub fn teleport_targets(pos: &Position, us: Color) -> Bitboard {
    let t = &*TABLES;
    let them = us.flip();
    let empty = pos.empty_squares();
    let mut targets = 0u64;

    let own_king = pos.king_sq(us);
    let their_king = pos.king_sq(them);

    // 1. Both king rings.
    if let Some(k) = own_king {
        targets |= t.ring[k];
    }
    if let Some(k) = their_king {
        targets |= t.ring[k];
    }

    // 2. Interposition squares against a slider that can currently reach our
    //    king's square (landing there detonates the 3x3 and kills the king).
    if let Some(k) = own_king {
        for pt in [PieceType::Rook, PieceType::Bishop] {
            for s in bits(pos.bb_of(them, pt)) {
                if slider_attacks(pt, s, pos.occ) & bb(k) != 0 {
                    targets |= t.between[s][k];
                }
            }
        }
    }

    // 3. Squares one of our own rooks/bishops can reach, whose 3x3 blast would
    //    catch an enemy non-pawn. Teleporting a pawn there manufactures the
    //    detonator (spec §5.2).
    let mut reach = 0u64;
    for pt in [PieceType::Rook, PieceType::Bishop] {
        for s in bits(pos.bb_of(us, pt)) {
            reach |= slider_attacks(pt, s, pos.occ);
        }
    }
    let enemy_worthwhile = pos.occ_color[them.index()] & !pos.bb_of(them, PieceType::Pawn);
    if enemy_worthwhile != 0 {
        for c in bits(reach & empty) {
            if t.ring[c] & enemy_worthwhile != 0 {
                targets |= bb(c);
            }
        }
    }

    targets &= empty;

    // 4. Parking squares: quiet, away from both kings.
    let mut quiet = empty & !targets;
    if let Some(k) = own_king {
        quiet &= !t.blast[k];
    }
    if let Some(k) = their_king {
        quiet &= !t.blast[k];
    }
    targets | sample_spread(quiet, 4)
}

/// Pick up to `n` set bits spread evenly through `mask`. Deterministic — the
/// engine must be reproducible for gating and perft-style diffing.
fn sample_spread(mask: Bitboard, n: usize) -> Bitboard {
    let total = mask.count_ones() as usize;
    if total == 0 || n == 0 {
        return 0;
    }
    if total <= n {
        return mask;
    }
    let step = total / n;
    let mut out = 0u64;
    let mut taken = 0;
    for (i, s) in bits(mask).enumerate() {
        if i % step == 0 && taken < n {
            out |= bb(s);
            taken += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Exact capture effects — spec §5.6
// ---------------------------------------------------------------------------

/// What `mv` would destroy, without playing it: `(captured_mask, mover_dies)`.
/// The captured mask excludes the mover itself.
///
/// Standard SEE is meaningless in this variant because there is no recapture
/// sequence; the true swing of any move is one mask intersection away, which is
/// strictly better information than a real engine has at ordering time.
#[inline]
pub fn capture_effects(pos: &Position, mv: Move) -> (Bitboard, bool) {
    let (from, to) = (mv.from(), mv.to());
    let t = &*TABLES;
    let occ = pos.occ;
    let p = match pos.piece_at(from) {
        Some(p) => p,
        None => return (0, false),
    };
    match piece_type_of(p) {
        PieceType::Pawn => (0, false),
        PieceType::Knight => (t.knight_path_lookup[from][to] & occ, false),
        PieceType::Rook | PieceType::Bishop => {
            if occ & bb(to) != 0 {
                (t.blast[to] & occ & !bb(from), true)
            } else {
                (0, false)
            }
        }
        PieceType::Queen => ((t.between[from][to] | bb(to)) & occ, false),
        PieceType::King => (bb(to) & occ, false),
    }
}

#[inline(always)]
pub fn is_capture(pos: &Position, mv: Move) -> bool {
    let (cap, died) = capture_effects(pos, mv);
    cap != 0 || died
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    #[test]
    fn startpos_teleport_count_is_pawns_times_empties() {
        let pos = Position::startpos();
        let moves = generate_vec(&pos, GenMode::All);
        let pawn_moves = moves
            .iter()
            .filter(|m| {
                pos.piece_at(m.from()).map(crate::types::piece_type_of) == Some(PieceType::Pawn)
            })
            .count();
        let expected =
            pos.bb_of(Color::White, PieceType::Pawn).count_ones() as usize * (64 - 32);
        assert_eq!(pawn_moves, expected);
    }

    #[test]
    fn queen_ray_enumeration_matches_spec_example() {
        // Spec §2.4: a ray with pieces at distance 3 and 6 yields destinations
        // {1,2,3,4,5,6} and nothing beyond.
        // Queen on b1, blockers on b4 (distance 3) and b7 (distance 6).
        let pos = Position::from_fen("8/1p6/8/8/1p6/8/8/1Q2k2K w - - 0 1").unwrap();
        let moves = generate_vec(&pos, GenMode::All);
        let b1 = parse_square("b1").unwrap();
        let north: Vec<String> = moves
            .iter()
            .filter(|m| m.from() == b1 && file_of(m.to()) == 1 && m.to() > b1)
            .map(|m| square_name(m.to()))
            .collect();
        let mut got = north.clone();
        got.sort();
        assert_eq!(got, vec!["b2", "b3", "b4", "b5", "b6", "b7"]);
        assert!(!got.contains(&"b8".to_string()));
        let _ = pos;
    }

    #[test]
    fn capture_effects_agrees_with_make_move() {
        // Property check across a random walk: the ordering-time capture mask must
        // equal what make_move actually destroys, or move ordering lies.
        let mut rng = SplitMix64::new(12345);
        for _ in 0..300 {
            let mut pos = Position::startpos();
            for _ in 0..40 {
                if pos.result().is_some() {
                    break;
                }
                let moves = generate_vec(&pos, GenMode::All);
                for &m in moves.iter() {
                    let (cap, died) = capture_effects(&pos, m);
                    let (_, info) = pos.after(m);
                    assert_eq!(
                        cap,
                        info.captured,
                        "capture mask mismatch for {m} in {}",
                        pos.to_fen()
                    );
                    assert_eq!(died, info.mover_died, "mover_died mismatch for {m}");
                }
                let m = moves[rng.below(moves.len())];
                pos.make_move(m);
                pos.validate().unwrap();
            }
        }
    }

    #[test]
    fn restricted_generation_is_a_subset_of_full() {
        let mut rng = SplitMix64::new(777);
        let mut pos = Position::startpos();
        for _ in 0..60 {
            if pos.result().is_some() {
                break;
            }
            let all = generate_vec(&pos, GenMode::All);
            for m in generate_vec(&pos, GenMode::Search) {
                assert!(all.contains(&m), "{m} not in full generation");
            }
            for m in generate_vec(&pos, GenMode::Captures) {
                assert!(all.contains(&m), "capture {m} not in full generation");
                assert!(is_capture(&pos, m), "{m} generated as capture but isn't one");
            }
            let m = all[rng.below(all.len())];
            pos.make_move(m);
        }
    }

    #[test]
    fn move_list_capacity_has_real_headroom() {
        // `push` drops silently past MAX_MOVES in release builds, so the bound
        // needs to be genuinely unreachable, not merely usually enough. The
        // theoretical worst case is a full 16-piece side against a bare king:
        // 8 pawns x 47 empty squares + ~105 piece moves = 481.
        let worst = Position::from_fen("7k/8/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1").unwrap();
        let n = generate_vec(&worst, GenMode::All).len();
        assert!(n < MAX_MOVES, "{n} moves against a capacity of {MAX_MOVES}");

        let mut peak = n;
        let mut rng = SplitMix64::new(0xB0A7);
        for _ in 0..400 {
            let mut pos = Position::startpos();
            for _ in 0..60 {
                if pos.result().is_some() {
                    break;
                }
                let moves = generate_vec(&pos, GenMode::All);
                peak = peak.max(moves.len());
                pos.make_move(moves[rng.below(moves.len())]);
            }
        }
        assert!(
            peak < MAX_MOVES,
            "observed {peak} moves against a capacity of {MAX_MOVES}"
        );
    }

    #[test]
    fn every_position_has_a_move() {
        // Spec §1.4 — the king always has somewhere to go.
        let mut rng = SplitMix64::new(31337);
        for _ in 0..200 {
            let mut pos = Position::startpos();
            for _ in 0..80 {
                if pos.result().is_some() {
                    break;
                }
                let moves = generate_vec(&pos, GenMode::All);
                assert!(!moves.is_empty(), "no moves in {}", pos.to_fen());
                assert!(
                    !generate_vec(&pos, GenMode::Search).is_empty(),
                    "no search moves in {}",
                    pos.to_fen()
                );
                pos.make_move(moves[rng.below(moves.len())]);
            }
        }
    }
}
