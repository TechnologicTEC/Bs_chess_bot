//! Training-sample format (build plan, Phase 5).
//!
//! One record is a fixed 32 bytes: the position, the search score that produced
//! it, and the eventual game result. Fixed-width so the Python trainer can
//! `np.fromfile` the whole shard in one call and so shards concatenate with `cat`.
//!
//! Layout (little-endian):
//!
//! | Offset | Size | Field                                                   |
//! |--------|------|---------------------------------------------------------|
//! | 0      | 8    | occupancy bitboard                                      |
//! | 8      | 16   | 4 bits per occupied square, ascending; piece index 0..11 |
//! | 24     | 1    | side to move (0 = white)                                |
//! | 25     | 1    | ply                                                     |
//! | 26     | 2    | search score, centipawns, side-to-move relative          |
//! | 28     | 1    | game result from White's view: +1 / 0 / -1              |
//! | 29     | 1    | flags: bit 0 = best move was quiet                       |
//! | 30     | 2    | reserved                                                |

use crate::position::{GameResult, Position};
use crate::types::*;

pub const RECORD_SIZE: usize = 32;
pub const FLAG_QUIET: u8 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sample {
    pub occ: Bitboard,
    pub nibbles: [u8; 16],
    pub side: u8,
    pub ply: u8,
    pub score: i16,
    pub result: i8,
    pub flags: u8,
}

impl Sample {
    pub fn from_position(pos: &Position, score: i32, quiet: bool) -> Sample {
        debug_assert!(pos.occ.count_ones() <= 32, "more than 32 pieces on the board");
        let mut nibbles = [0u8; 16];
        for (i, s) in bits(pos.occ).enumerate() {
            let p = pos.piece_at(s).expect("occupancy disagrees with piece boards") as u8;
            if i % 2 == 0 {
                nibbles[i / 2] |= p;
            } else {
                nibbles[i / 2] |= p << 4;
            }
        }
        Sample {
            occ: pos.occ,
            nibbles,
            side: pos.side.index() as u8,
            ply: pos.ply.min(255) as u8,
            score: score.clamp(-30_000, 30_000) as i16,
            result: 0,
            flags: if quiet { FLAG_QUIET } else { 0 },
        }
    }

    pub fn set_result(&mut self, r: GameResult) {
        self.result = match r {
            GameResult::Draw => 0,
            GameResult::Win(Color::White) => 1,
            GameResult::Win(Color::Black) => -1,
        };
    }

    pub fn to_position(&self) -> Position {
        let mut pos = Position::empty();
        for (i, s) in bits(self.occ).enumerate() {
            let byte = self.nibbles[i / 2];
            let p = if i % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            pos.put_piece(p as usize, s);
        }
        pos.side = Color::from_index(self.side as usize);
        pos.ply = self.ply as u16;
        pos.recompute_key();
        pos
    }

    pub fn encode(&self) -> [u8; RECORD_SIZE] {
        let mut b = [0u8; RECORD_SIZE];
        b[0..8].copy_from_slice(&self.occ.to_le_bytes());
        b[8..24].copy_from_slice(&self.nibbles);
        b[24] = self.side;
        b[25] = self.ply;
        b[26..28].copy_from_slice(&self.score.to_le_bytes());
        b[28] = self.result as u8;
        b[29] = self.flags;
        b
    }

    pub fn decode(b: &[u8]) -> Option<Sample> {
        if b.len() < RECORD_SIZE {
            return None;
        }
        let mut occ = [0u8; 8];
        occ.copy_from_slice(&b[0..8]);
        let mut nibbles = [0u8; 16];
        nibbles.copy_from_slice(&b[8..24]);
        Some(Sample {
            occ: u64::from_le_bytes(occ),
            nibbles,
            side: b[24],
            ply: b[25],
            score: i16::from_le_bytes([b[26], b[27]]),
            result: b[28] as i8,
            flags: b[29],
        })
    }

    #[inline]
    pub fn is_quiet(&self) -> bool {
        self.flags & FLAG_QUIET != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::{generate_vec, GenMode};
    use crate::tables::SplitMix64;

    #[test]
    fn record_is_32_bytes() {
        assert_eq!(Sample::from_position(&Position::startpos(), 0, true).encode().len(), 32);
    }

    #[test]
    fn roundtrips_through_random_play() {
        let mut rng = SplitMix64::new(2024);
        for _ in 0..200 {
            let mut pos = Position::startpos();
            for _ in 0..rng.below(60) {
                if pos.result().is_some() {
                    break;
                }
                let moves = generate_vec(&pos, GenMode::All);
                pos.make_move(moves[rng.below(moves.len())]);
            }
            let mut s = Sample::from_position(&pos, -123, true);
            s.set_result(GameResult::Win(Color::Black));
            let back = Sample::decode(&s.encode()).unwrap();
            assert_eq!(back, s);
            let rebuilt = back.to_position();
            assert_eq!(rebuilt.pieces, pos.pieces, "{}", pos.to_fen());
            assert_eq!(rebuilt.side, pos.side);
            assert_eq!(rebuilt.key, pos.key);
            assert_eq!(back.score, -123);
            assert_eq!(back.result, -1);
        }
    }
}
