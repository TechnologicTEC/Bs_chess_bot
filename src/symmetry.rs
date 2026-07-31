//! Dihedral symmetry (build plan, Phase 3).
//!
//! No castling and no directional pawn movement means the rules are fully
//! invariant under the dihedral group of the square. Two consequences:
//!
//! 1. Colour canonicalisation is free — swap piece colours and flip the side to
//!    move. No vertical board flip is needed, because no piece has a forward
//!    direction.
//! 2. Every training position can be augmented 8x (4 rotations x 2 mirrors).
//!
//! A bug in here is nearly invisible downstream: it just makes training
//! mysteriously ineffective. So the transforms are unit-tested against movegen,
//! `make_move` and evaluation rather than against themselves.

use crate::position::Position;
use crate::types::*;
use std::sync::LazyLock;

pub const NUM_SYMMETRIES: usize = 8;

pub const SYMMETRY_NAMES: [&str; NUM_SYMMETRIES] = [
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "mirror_files",
    "mirror_ranks",
    "diagonal",
    "antidiagonal",
];

/// `(rank, file) -> (rank, file)` for each group element.
fn apply_coords(i: usize, r: usize, c: usize) -> (usize, usize) {
    match i {
        0 => (r, c),         // identity
        1 => (c, 7 - r),     // rotate 90
        2 => (7 - r, 7 - c), // rotate 180
        3 => (7 - c, r),     // rotate 270
        4 => (r, 7 - c),     // mirror across the vertical axis
        5 => (7 - r, c),     // mirror across the horizontal axis
        6 => (c, r),         // transpose (a1-h8 diagonal)
        7 => (7 - c, 7 - r), // anti-transpose (a8-h1 diagonal)
        _ => unreachable!("symmetry index out of range"),
    }
}

pub struct SymmetryTables {
    /// `perm[i][s]` is where square `s` lands under symmetry `i`.
    pub perm: [[u8; 64]; NUM_SYMMETRIES],
    /// Index of the symmetry that undoes `i`.
    pub inverse: [usize; NUM_SYMMETRIES],
}

fn build() -> SymmetryTables {
    let mut perm = [[0u8; 64]; NUM_SYMMETRIES];
    for (i, row) in perm.iter_mut().enumerate() {
        for (s, slot) in row.iter_mut().enumerate() {
            let (r, c) = apply_coords(i, rank_of(s), file_of(s));
            *slot = sq(r, c) as u8;
        }
    }
    // Find each element's inverse by brute force — the group has eight members.
    let mut inverse = [0usize; NUM_SYMMETRIES];
    for (i, inv) in inverse.iter_mut().enumerate() {
        for (j, candidate) in perm.iter().enumerate() {
            if (0..64).all(|s| candidate[perm[i][s] as usize] as usize == s) {
                *inv = j;
                break;
            }
        }
    }
    SymmetryTables { perm, inverse }
}

pub static SYM: LazyLock<SymmetryTables> = LazyLock::new(build);

#[inline]
pub fn transform_square(i: usize, s: usize) -> usize {
    SYM.perm[i][s] as usize
}

pub fn transform_bb(i: usize, b: Bitboard) -> Bitboard {
    if i == 0 {
        return b;
    }
    let p = &SYM.perm[i];
    let mut out = 0u64;
    for s in bits(b) {
        out |= bb(p[s] as usize);
    }
    out
}

pub fn transform_move(i: usize, m: Move) -> Move {
    Move::new(transform_square(i, m.from()), transform_square(i, m.to()))
}

pub fn transform_position(i: usize, pos: &Position) -> Position {
    if i == 0 {
        return *pos;
    }
    let mut out = Position::empty();
    for p in 0..NUM_PIECES {
        out.pieces[p] = transform_bb(i, pos.pieces[p]);
    }
    out.occ_color[0] = transform_bb(i, pos.occ_color[0]);
    out.occ_color[1] = transform_bb(i, pos.occ_color[1]);
    out.occ = transform_bb(i, pos.occ);
    out.side = pos.side;
    out.ply = pos.ply;
    out.recompute_key();
    out
}

/// Swap piece colours and the side to move. Because no piece has a forward
/// direction, this alone canonicalises the position — no board flip required.
pub fn swap_colors(pos: &Position) -> Position {
    let mut out = Position::empty();
    for p in 0..NUM_PIECES {
        let pt = piece_type_of(p);
        let c = piece_color(p).flip();
        out.pieces[piece_index(c, pt)] = pos.pieces[p];
    }
    out.occ_color[0] = pos.occ_color[1];
    out.occ_color[1] = pos.occ_color[0];
    out.occ = pos.occ;
    out.side = pos.side.flip();
    out.ply = pos.ply;
    out.recompute_key();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::evaluate_white;
    use crate::movegen::{generate_vec, GenMode};
    use crate::tables::SplitMix64;
    use std::collections::HashSet;

    fn random_positions(n: usize, seed: u64) -> Vec<Position> {
        let mut rng = SplitMix64::new(seed);
        let mut out = Vec::new();
        while out.len() < n {
            let mut pos = Position::startpos();
            let depth = 1 + rng.below(50);
            for _ in 0..depth {
                if pos.result().is_some() {
                    break;
                }
                let moves = generate_vec(&pos, GenMode::All);
                pos.make_move(moves[rng.below(moves.len())]);
            }
            if pos.result().is_none() {
                out.push(pos);
            }
        }
        out
    }

    #[test]
    fn permutations_are_bijections_and_form_a_group() {
        for (i, name) in SYMMETRY_NAMES.iter().enumerate() {
            let seen: HashSet<u8> = SYM.perm[i].iter().copied().collect();
            assert_eq!(seen.len(), 64, "{name} is not a bijection");
        }
        // Closure: composing any two elements yields another element.
        for (i, a) in SYMMETRY_NAMES.iter().enumerate() {
            for (j, b) in SYMMETRY_NAMES.iter().enumerate() {
                let composed: Vec<u8> =
                    (0..64).map(|s| SYM.perm[j][SYM.perm[i][s] as usize]).collect();
                assert!(
                    SYM.perm.iter().any(|p| p.as_slice() == composed.as_slice()),
                    "{b} o {a} left the group"
                );
            }
        }
        // Inverses round-trip.
        for i in 0..NUM_SYMMETRIES {
            for s in 0..64 {
                assert_eq!(transform_square(SYM.inverse[i], transform_square(i, s)), s);
            }
        }
    }

    #[test]
    fn movegen_commutes_with_symmetry() {
        for pos in random_positions(40, 4242) {
            let base: HashSet<Move> = generate_vec(&pos, GenMode::All).into_iter().collect();
            for (i, name) in SYMMETRY_NAMES.iter().enumerate() {
                let tpos = transform_position(i, &pos);
                assert!(tpos.validate().is_ok());
                let got: HashSet<Move> = generate_vec(&tpos, GenMode::All).into_iter().collect();
                let want: HashSet<Move> =
                    base.iter().map(|&m| transform_move(i, m)).collect();
                assert_eq!(got, want, "{name} broke movegen on {}", pos.to_fen());
            }
        }
    }

    #[test]
    fn make_move_commutes_with_symmetry() {
        let mut rng = SplitMix64::new(99);
        for pos in random_positions(30, 8181) {
            let moves = generate_vec(&pos, GenMode::All);
            for _ in 0..8 {
                let m = moves[rng.below(moves.len())];
                let (child, _) = pos.after(m);
                for (i, name) in SYMMETRY_NAMES.iter().enumerate() {
                    let (tchild, _) = transform_position(i, &pos).after(transform_move(i, m));
                    let expect = transform_position(i, &child);
                    assert_eq!(
                        tchild.pieces, expect.pieces,
                        "{name} broke make_move for {m} on {}",
                        pos.to_fen()
                    );
                }
            }
        }
    }

    #[test]
    fn evaluation_is_symmetry_invariant() {
        for pos in random_positions(40, 1234) {
            let base = evaluate_white(&pos);
            for (i, name) in SYMMETRY_NAMES.iter().enumerate() {
                assert_eq!(
                    evaluate_white(&transform_position(i, &pos)),
                    base,
                    "{name} changed the evaluation of {}",
                    pos.to_fen()
                );
            }
        }
    }

    #[test]
    fn colour_swap_negates_and_is_an_involution() {
        for pos in random_positions(40, 5150) {
            let swapped = swap_colors(&pos);
            assert!(swapped.validate().is_ok());
            assert_eq!(evaluate_white(&swapped), -evaluate_white(&pos));
            let back = swap_colors(&swapped);
            assert_eq!(back.pieces, pos.pieces);
            assert_eq!(back.side, pos.side);
            assert_eq!(back.key, pos.key);
        }
    }

    #[test]
    fn colour_swap_maps_moves_one_to_one() {
        for pos in random_positions(25, 606) {
            let ours: HashSet<Move> = generate_vec(&pos, GenMode::All).into_iter().collect();
            let theirs: HashSet<Move> = generate_vec(&swap_colors(&pos), GenMode::All)
                .into_iter()
                .collect();
            assert_eq!(ours, theirs, "colour swap changed the move set");
        }
    }
}
