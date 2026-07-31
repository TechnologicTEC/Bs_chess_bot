//! # Bullshit Chess engine
//!
//! An engine for the variant defined in `bullshit-chess-spec.md`, built to the
//! plan in `bullshit-chess-build-plan.md`.
//!
//! Module map, by build-plan phase:
//!
//! | Phase | Modules                                        |
//! |-------|------------------------------------------------|
//! | 1     | [`types`], [`tables`], [`position`], [`movegen`], [`perft`] |
//! | 2     | [`eval`], [`search`]                            |
//! | 3     | [`symmetry`], [`selfplay`], `bin/bschess` (operator mode) |
//! | 4     | [`nnue`]                                        |
//! | 5     | [`data`], `tools/train.py`                      |
//! | 6     | [`gauntlet`]                                    |
//!
//! The one invariant worth internalising: **every pseudo-legal move is legal**
//! (spec §1.1) and **a side to move always has one** (spec §1.4). There is no
//! check, no pin detection, no checkmate search and no stalemate.

pub mod data;
pub mod eval;
pub mod gauntlet;
pub mod movegen;
pub mod nnue;
pub mod perft;
pub mod position;
pub mod search;
pub mod selfplay;
pub mod symmetry;
pub mod tables;
pub mod types;

/// Build the lookup tables up front so the first search does not pay for them
/// inside its own clock.
pub fn init() {
    tables::init();
}

pub use eval::Evaluator;
pub use movegen::{generate, generate_vec, GenMode, MoveList};
pub use position::{GameResult, Position};
pub use search::{best_move, SearchLimits, SearchOptions, SearchResult, Searcher};
pub use types::{Color, Move, PieceType};
