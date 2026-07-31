//! Perft â€” the Phase 1 gate.
//!
//! **Definition** (must match `ref/reference.py` exactly):
//!
//! ```text
//! perft(pos, 0) = 1
//! perft(pos, d) = 0                                    if pos is terminal (spec Â§3)
//! perft(pos, d) = sum over legal moves of perft(child, d-1)   otherwise
//! ```
//!
//! Terminal positions contribute nothing at `d > 0` because no move may be played
//! from them. Always use `GenMode::All` â€” restricted teleport generation is a
//! search heuristic, not a rule.

use crate::movegen::{generate, GenMode, MoveList};
use crate::position::Position;
use crate::types::Move;

pub fn perft(pos: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    if pos.result().is_some() {
        return 0;
    }
    let mut list = MoveList::new();
    generate(pos, GenMode::All, &mut list);
    if depth == 1 {
        // Bulk counting is only valid when every child is a leaf; children that
        // are terminal still count as reached positions at depth 1.
        return list.len() as u64;
    }
    let mut total = 0;
    for &mv in list.as_slice() {
        let (child, _) = pos.after(mv);
        total += perft(&child, depth - 1);
    }
    total
}

/// Per-root-move breakdown, sorted, for diffing against the reference.
pub fn perft_divide(pos: &Position, depth: u32) -> Vec<(Move, u64)> {
    let mut list = MoveList::new();
    generate(pos, GenMode::All, &mut list);
    let mut out: Vec<(Move, u64)> = list
        .as_slice()
        .iter()
        .map(|&mv| {
            let (child, _) = pos.after(mv);
            (mv, if depth <= 1 { 1 } else { perft(&child, depth - 1) })
        })
        .collect();
    out.sort_by_key(|(m, _)| m.to_string());
    out
}

/// Parallel perft for the deeper checks.
pub fn perft_parallel(pos: &Position, depth: u32) -> u64 {
    use rayon::prelude::*;
    if depth <= 2 {
        return perft(pos, depth);
    }
    if pos.result().is_some() {
        return 0;
    }
    let mut list = MoveList::new();
    generate(pos, GenMode::All, &mut list);
    list.as_slice()
        .par_iter()
        .map(|&mv| {
            let (child, _) = pos.after(mv);
            perft(&child, depth - 1)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_one_equals_the_move_count() {
        let pos = Position::startpos();
        let mut list = MoveList::new();
        generate(&pos, GenMode::All, &mut list);
        assert_eq!(perft(&pos, 1), list.len() as u64);
    }

    #[test]
    fn parallel_matches_serial() {
        let pos = Position::startpos();
        for d in 1..=3 {
            assert_eq!(perft(&pos, d), perft_parallel(&pos, d), "depth {d}");
        }
    }

    #[test]
    fn divide_sums_to_perft() {
        let pos = Position::startpos();
        for d in 1..=3 {
            let total: u64 = perft_divide(&pos, d).iter().map(|(_, n)| n).sum();
            assert_eq!(total, perft(&pos, d), "depth {d}");
        }
    }
}
