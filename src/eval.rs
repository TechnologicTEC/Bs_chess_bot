//! Hand-written evaluation — spec §6, with the strategic consequences of §5
//! baked in as positional terms.
//!
//! All scores are centipawns from White's point of view internally; `evaluate`
//! returns them from the side to move's point of view for negamax.

use crate::movegen::capture_effects;
use crate::nnue::Network;
use crate::position::Position;
use crate::tables::*;
use crate::types::*;

/// Spec §6 starting weights, x100. Tune by self-play afterwards.
///
/// | Piece  | Value | Why                                    |
/// |--------|-------|----------------------------------------|
/// | Queen  | 10.0  | survives, 2 captures, long range       |
/// | Knight |  9.0  | survives, 3 captures, unblockable      |
/// | Rook   |  4.5  | consumable: line control + threat       |
/// | Bishop |  4.0  | consumable, lands on one colour only    |
/// | Pawn   |  0.25 | blocker, detonator primer, draw resource|
pub const PIECE_VALUE: [i32; NUM_PIECE_TYPES] = [
    25,   // Pawn
    900,  // Knight
    400,  // Bishop
    450,  // Rook
    1000, // Queen
    0,    // King — terminal, never counted as material
];

/// Used only for move ordering, so that king captures sort ahead of everything.
pub const KING_ORDER_VALUE: i32 = 20_000;

// --- positional weights (spec §6) -------------------------------------------

/// Per occupied square in the king's 3x3 that an enemy rook/bishop can land on
/// this move. That is an immediate loss threat, not a soft weakness.
pub const W_LIVE_DETONATOR: i32 = -300;
/// Per occupied square adjacent to the own king generally — a latent detonator.
/// Spec §5.1: pieces near your own king are a liability, not a shield.
pub const W_LATENT_DETONATOR: i32 = -40;
/// Per enemy piece in the best blast we can currently deliver, when that blast
/// nets at least two.
pub const W_CLUSTER: i32 = 30;
/// Per enemy knight whose sweep path already crosses our king square. The sweep
/// cannot be blocked (spec §5.4), so proximity alone is dangerous.
pub const W_KNIGHT_ON_KING: i32 = -120;
pub const W_TEMPO: i32 = 15;

/// Total heavy material at the start of a game, used to scale the material
/// advantage down as the position drifts toward the kings-and-pawns draw.
const HEAVY_AT_START: i32 = 2 * (2 * 900 + 2 * 400 + 2 * 450 + 1000);

/// Score bounds. Wins are stored as `MATE - ply` so that shorter wins are preferred.
pub const MATE: i32 = 30_000;
pub const MATE_THRESHOLD: i32 = MATE - 1000;
pub const INFINITY: i32 = 32_000;

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
pub enum Evaluator {
    /// Phase 2 hand evaluation. Also generation 0 of the training loop.
    #[default]
    Hand,
    /// Phase 4 NNUE.
    Nnue(std::sync::Arc<Network>),
}

impl Evaluator {
    #[inline]
    pub fn eval(&self, pos: &Position) -> i32 {
        match self {
            Evaluator::Hand => evaluate(pos),
            Evaluator::Nnue(n) => n.evaluate(pos),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Evaluator::Hand => "hand",
            Evaluator::Nnue(_) => "nnue",
        }
    }
}

// ---------------------------------------------------------------------------
// Hand evaluation
// ---------------------------------------------------------------------------

/// Returns centipawns from the side to move's point of view.
pub fn evaluate(pos: &Position) -> i32 {
    let white = evaluate_white(pos);
    if pos.side == Color::White {
        white + W_TEMPO
    } else {
        -white + W_TEMPO
    }
}

pub fn evaluate_white(pos: &Position) -> i32 {
    let mat_w = pos.material_cp(Color::White);
    let mat_b = pos.material_cp(Color::Black);

    // Spec §6: scale the advantage down as heavy material drains away, because
    // the kings-and-pawns draw becomes the trailing side's live resource (§5.5).
    let heavy = heavy_material_cp(pos);
    let scale = 35 + 65 * heavy.min(HEAVY_AT_START) / HEAVY_AT_START;
    let material = (mat_w - mat_b) * scale / 100;

    material + king_safety(pos, Color::White) - king_safety(pos, Color::Black)
        + cluster_bonus(pos, Color::White)
        - cluster_bonus(pos, Color::Black)
}

fn heavy_material_cp(pos: &Position) -> i32 {
    let mut total = 0;
    for c in [Color::White, Color::Black] {
        for pt in [
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ] {
            total += pos.bb_of(c, pt).count_ones() as i32 * PIECE_VALUE[pt.index()];
        }
    }
    total
}

/// Spec §5.1 inverted-intuition term: every occupied square in the king's 3x3 is
/// a live detonator, including our own pieces.
fn king_safety(pos: &Position, us: Color) -> i32 {
    let t = &*TABLES;
    let them = us.flip();
    let Some(k) = pos.king_sq(us) else {
        return 0;
    };

    let mut score = 0;

    // Latent: anything standing next to the king.
    let ring_occ = t.ring[k] & pos.occ;
    score += W_LATENT_DETONATOR * ring_occ.count_ones() as i32;

    // Live: an enemy rook or bishop that can land on any occupied square of the
    // king's 3x3 — including the king's own square — kills the king outright.
    let detonation_points = t.blast[k] & pos.occ;
    let mut live = 0u64;
    for pt in [PieceType::Rook, PieceType::Bishop] {
        for s in bits(pos.bb_of(them, pt)) {
            live |= slider_attacks(pt, s, pos.occ) & detonation_points;
        }
    }
    score += W_LIVE_DETONATOR * live.count_ones() as i32;

    // Unblockable knight sweeps already crossing the king square.
    for s in bits(pos.bb_of(them, PieceType::Knight)) {
        if t.knight_span[s] & bb(k) != 0 {
            score += W_KNIGHT_ON_KING;
        }
    }

    score
}

/// Spec §6: reward enemy pieces clustered where one of our detonations nets two
/// or more. Uses the best single blast rather than summing over all of them, so
/// the same cluster is not counted once per attacking slider.
fn cluster_bonus(pos: &Position, us: Color) -> i32 {
    let t = &*TABLES;
    let them = us.flip();
    let enemy = pos.occ_color[them.index()];
    let mut best = 0u32;
    for pt in [PieceType::Rook, PieceType::Bishop] {
        for s in bits(pos.bb_of(us, pt)) {
            // Detonation requires landing on an occupied square.
            for to in bits(slider_attacks(pt, s, pos.occ) & pos.occ) {
                let hit = t.blast[to] & enemy;
                best = best.max(hit.count_ones());
            }
        }
    }
    if best >= 2 {
        W_CLUSTER * best as i32
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Exact move delta — spec §5.6, used for move ordering
// ---------------------------------------------------------------------------

#[inline]
fn order_value(p: usize) -> i32 {
    let pt = piece_type_of(p);
    if pt == PieceType::King {
        KING_ORDER_VALUE
    } else {
        PIECE_VALUE[pt.index()]
    }
}

/// True material swing of `mv` in centipawns, from the mover's point of view.
///
/// There is no recapture sequence in this variant, so this single mask
/// intersection *is* the exact exchange value — not a static approximation.
pub fn move_delta(pos: &Position, mv: Move) -> i32 {
    let (captured, mover_died) = capture_effects(pos, mv);
    let us = pos.side;
    let mut delta = 0;

    if captured != 0 {
        for p in 0..NUM_PIECES {
            let hit = pos.pieces[p] & captured;
            if hit != 0 {
                let v = order_value(p) * hit.count_ones() as i32;
                if piece_color(p) == us {
                    delta -= v;
                } else {
                    delta += v;
                }
            }
        }
    }
    if mover_died {
        if let Some(p) = pos.piece_at(mv.from()) {
            delta -= order_value(p);
        }
    }
    delta
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_is_balanced() {
        let pos = Position::startpos();
        assert_eq!(evaluate_white(&pos), 0);
    }

    #[test]
    fn evaluation_is_colour_symmetric() {
        // Same position with colours swapped must evaluate to the negation.
        let pos = Position::from_fen("4k3/8/8/3n4/8/8/4R3/4K3 w - - 0 1").unwrap();
        let mirrored = crate::symmetry::swap_colors(&pos);
        assert_eq!(evaluate_white(&pos), -evaluate_white(&mirrored));
    }

    #[test]
    fn empty_ring_is_safer_than_crowded_ring() {
        // Spec §5.1: friendly pieces around your own king are a liability.
        let bare = Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let crowded = Position::from_fen("4k3/8/8/8/8/8/3PPP2/R3K3 w - - 0 1").unwrap();
        assert!(
            king_safety(&crowded, Color::White) < king_safety(&bare, Color::White),
            "crowding the king must score worse"
        );
    }

    #[test]
    fn detonation_delta_counts_the_mover() {
        // White rook a1 takes on a2 where a black knight stands; blast clears
        // a1..b3 region. Rook dies with it.
        let pos = Position::from_fen("4k3/8/8/8/8/8/n7/R3K3 w - - 0 1").unwrap();
        let mv = Move::parse("a1a2").unwrap();
        // +knight (900) - rook (450) = 450.
        assert_eq!(move_delta(&pos, mv), 900 - 450);
    }

    #[test]
    fn king_capture_dominates_ordering() {
        let pos = Position::from_fen("8/8/8/8/8/8/4k3/4K3 w - - 0 1").unwrap();
        let mv = Move::parse("e1e2").unwrap();
        assert_eq!(move_delta(&pos, mv), KING_ORDER_VALUE);
    }
}
