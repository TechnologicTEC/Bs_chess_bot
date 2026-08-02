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

/// Spec §6 piece values, x100.
///
/// **These are rule values, not evaluation preferences — do not tune them.**
/// Spec §4 defines the ply-cap adjudication in terms of them, so they are part of
/// how a game is *scored*, and both engines must agree. Changing them here would
/// silently make this engine adjudicate capped games differently from an opponent
/// implementing the same spec — exactly the divergence §8 warns about.
///
/// The tunable *evaluation* weights live in `TUNED_WEIGHTS` below.
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

/// Weights in `eval_features` order. Spec §6's values are the starting guess;
/// `TUNED_WEIGHTS` is what fitting them to 759k labelled self-play positions
/// produced. Both are kept so the two can be played against each other.
pub type HandWeights = [i32; NUM_EVAL_FEATURES];

/// Build a weight vector from the ten original terms, leaving every added
/// feature at zero. A zero weight contributes nothing, so this reproduces the
/// pre-piece-square-table evaluation exactly — which is what makes it a valid
/// baseline to measure the new terms against.
const fn base_weights(core: [i32; 10]) -> HandWeights {
    let mut w = [0; NUM_EVAL_FEATURES];
    let mut i = 0;
    while i < 10 {
        w[i] = core[i];
        i += 1;
    }
    w
}

pub const SPEC_WEIGHTS: HandWeights =
    base_weights([25, 900, 400, 450, 1000, -40, -300, -120, 30, 15]);

/// Texel-tuned against 759k labelled positions from `data/gen1*.bin`.
/// Training loss 0.041183 -> 0.027392; **+49 ± 49 Elo over `SPEC_WEIGHTS`** at
/// equal time over 200 games, and +200 ± 80 against the best NNUE trained on the
/// same data. This is the default the engine plays with.
///
/// Three of the changes are worth knowing about:
///
/// * **Queen 10.0 -> 17.0.** Spec §6 reasoned "survives, 2 captures, long range"
///   and then under-scaled it badly. She is worth ~1.8 knights, not ~1.1.
/// * **Pawn 0.25 -> 0.70.** Manufacturing a detonator next to the enemy king
///   (§5.2) is worth far more than a quarter-pawn.
/// * **Latent detonator -0.40 -> +0.48, a sign flip.** See the note on
///   `king_safety_counts`.
pub const TUNED_WEIGHTS: HandWeights =
    base_weights([70, 944, 318, 456, 1696, 48, -321, -180, 69, 16]);

/// Refit including the piece-square tables and the three added terms, over 1.06M
/// labelled positions. Validation loss 0.026348 -> 0.016499, against the NNUE's
/// 0.0087 — so roughly half the remaining gap to what a 214k-parameter network
/// could extract, at no run-time cost.
///
/// Material and the tables are deliberately collinear (both scale with piece
/// count), so the individual numbers below are not separately meaningful — only
/// the total is. A knight reads as -157 material plus ~+680 table.
pub const TUNED_V2_WEIGHTS: HandWeights = [
    33, -157, 11, -112, 760, 8, -173, -12, 46, 9, //
    8, -11, -16, -16, 4, 6, 22, 2, -11, -2, // pawn table
    694, 647, 669, 688, 678, 692, 687, 677, 700, 715, // knight table
    278, 257, 267, 309, 340, 305, 418, 326, 458, 358, // bishop table
    430, 498, 451, 488, 480, 468, 472, 462, 469, 494, // rook table
    549, 520, 570, 395, 588, 553, 593, 590, 620, 683, // queen table
    -43, -48, -7, 59, -52, -46, -46, -80, -69, -88, // king table
    6, -10, 32, // knight span, attack primer, pawn endgame
];

/// Score bounds. Wins are stored as `MATE - ply` so that shorter wins are preferred.
pub const MATE: i32 = 30_000;
pub const MATE_THRESHOLD: i32 = MATE - 1000;
pub const INFINITY: i32 = 32_000;

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
pub enum Evaluator {
    /// The hand evaluation with tuned weights. This is what the engine plays.
    #[default]
    Hand,
    /// Candidate: the same evaluation refit with piece-square tables and three
    /// added terms. Promoted to `Hand` once it has beaten it over enough games.
    HandV2,
    /// The same evaluation with spec §6's starting guesses, kept as the baseline
    /// the tuned set is measured against.
    HandSpec,
    /// Phase 4 NNUE. Measured 200 Elo *behind* the tuned hand evaluation at equal
    /// time; kept for the record and for further experiments.
    Nnue(std::sync::Arc<Network>),
}

impl Evaluator {
    #[inline]
    pub fn eval(&self, pos: &Position) -> i32 {
        match self {
            Evaluator::Hand => evaluate_with(pos, &TUNED_WEIGHTS),
            Evaluator::HandV2 => evaluate_with(pos, &TUNED_V2_WEIGHTS),
            Evaluator::HandSpec => evaluate_with(pos, &SPEC_WEIGHTS),
            Evaluator::Nnue(n) => n.evaluate(pos),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Evaluator::Hand => "hand-tuned",
            Evaluator::HandV2 => "hand-v2",
            Evaluator::HandSpec => "hand-spec",
            Evaluator::Nnue(_) => "nnue",
        }
    }
}

// ---------------------------------------------------------------------------
// Hand evaluation
// ---------------------------------------------------------------------------

/// Returns centipawns from the side to move's point of view, using the tuned
/// weights — measured +49 Elo over the spec's starting guesses.
pub fn evaluate(pos: &Position) -> i32 {
    evaluate_with(pos, &TUNED_WEIGHTS)
}

pub fn evaluate_with(pos: &Position, w: &HandWeights) -> i32 {
    let white = evaluate_white_with(pos, w);
    if pos.side == Color::White {
        white + w[9]
    } else {
        -white + w[9]
    }
}

pub fn evaluate_white(pos: &Position) -> i32 {
    evaluate_white_with(pos, &TUNED_WEIGHTS)
}

/// Spec §6: scale the material advantage down as heavy material drains away,
/// because the kings-and-pawns draw becomes the trailing side's live resource
/// (§5.5). Returned as a percentage in `[35, 100]`.
///
/// Shared by `evaluate_white_with` and `eval_features` so the tuner cannot end up
/// optimising a slightly different function from the one the engine plays.
fn endgame_scale(pos: &Position, w: &HandWeights) -> i32 {
    let at_start = 2 * (2 * w[1] + 2 * w[2] + 2 * w[3] + w[4]);
    let mut current = 0;
    for c in [Color::White, Color::Black] {
        for (i, pt) in [
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ]
        .into_iter()
        .enumerate()
        {
            current += pos.bb_of(c, pt).count_ones() as i32 * w[i + 1];
        }
    }
    35 + 65 * current.min(at_start) / at_start.max(1)
}

/// Same arithmetic as the compiled-in evaluation, with the weights supplied.
/// Both the spec and tuned weight sets go through this one path, so an A/B
/// between them compares the numbers rather than two different code paths.
pub fn evaluate_white_with(pos: &Position, w: &HandWeights) -> i32 {
    let mut mat_w = 0;
    let mut mat_b = 0;
    for (i, pt) in [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
    ]
    .into_iter()
    .enumerate()
    {
        mat_w += pos.bb_of(Color::White, pt).count_ones() as i32 * w[i];
        mat_b += pos.bb_of(Color::Black, pt).count_ones() as i32 * w[i];
    }

    let scale = endgame_scale(pos, w);
    let material = (mat_w - mat_b) * scale / 100;

    let (lw, vw, kw) = king_safety_counts(pos, Color::White);
    let (lb, vb, kb) = king_safety_counts(pos, Color::Black);
    let safety = w[F_LATENT] * (lw - lb) + w[F_LIVE] * (vw - vb) + w[F_KNIGHT_ON_KING] * (kw - kb);
    let cluster =
        w[F_CLUSTER] * (cluster_count(pos, Color::White) - cluster_count(pos, Color::Black));

    // Piece-square tables. One table lookup and one add per piece on the board —
    // cheap enough that it does not move the node rate, which is the whole reason
    // to prefer this over a network.
    let orbit = &*ORBIT;
    let mut pst = 0;
    for p in 0..NUM_PIECES {
        let board = pos.pieces[p];
        if board == 0 {
            continue;
        }
        let base = F_PST + piece_type_of(p).index() * NUM_ORBITS;
        let sign = if piece_color(p) == Color::White { 1 } else { -1 };
        for s in bits(board) {
            pst += sign * w[base + orbit[s] as usize];
        }
    }

    let extra = w[F_KNIGHT_SPAN]
        * (knight_span_count(pos, Color::White) - knight_span_count(pos, Color::Black))
        + w[F_ATTACK_PRIMER]
            * (attack_primer_count(pos, Color::White) - attack_primer_count(pos, Color::Black))
        + w[F_PAWN_ENDGAME] * pawn_endgame_term(pos, scale);

    material + safety + cluster + pst + extra
}

/// Pawn-count difference weighted by how far the position has drained toward the
/// kings-and-pawns draw. Zero in the opening, largest in the endgame.
fn pawn_endgame_term(pos: &Position, scale: i32) -> i32 {
    let diff = pos.bb_of(Color::White, PieceType::Pawn).count_ones() as i32
        - pos.bb_of(Color::Black, PieceType::Pawn).count_ones() as i32;
    diff * (100 - scale) / 100
}


/// The raw counts behind the king-safety terms, kept separate so the tuner can
/// fit the weights without duplicating the logic they multiply.
/// Returns `(latent detonators, live detonators, enemy knights covering the king)`.
///
/// **Tuning contradicted spec §5.1 on the first of these.** The spec states, as a
/// headline result, that "pieces clustered near your own king are a liability,
/// not a shield", and §6 prices that at -0.4 per occupied neighbour. Fitted
/// against 759k labelled positions the term comes out at **+0.48** — a mild
/// asset, not a liability.
///
/// The *live* term, an enemy slider that can actually reach and detonate, stayed
/// strongly negative (-3.00 -> -3.21). So the refined reading is: being
/// reachable is what kills you; merely having pieces nearby does not.
///
/// One caveat worth keeping: the latent count includes pieces of both colours, so
/// it partially proxies for "I have more pieces on the board" and the regression
/// may be attributing some material advantage to it. The gauntlet says the tuned
/// set is worth +49 Elo overall, but it does not isolate this one term.
fn king_safety_counts(pos: &Position, us: Color) -> (i32, i32, i32) {
    let t = &*TABLES;
    let them = us.flip();
    let Some(k) = pos.king_sq(us) else {
        return (0, 0, 0);
    };

    // Latent: anything standing next to the king.
    let latent = (t.ring[k] & pos.occ).count_ones() as i32;

    // Live: an enemy rook or bishop that can land on any occupied square of the
    // king's 3x3 — including the king's own square — kills the king outright.
    let detonation_points = t.blast[k] & pos.occ;
    let mut live = 0u64;
    for pt in [PieceType::Rook, PieceType::Bishop] {
        for s in bits(pos.bb_of(them, pt)) {
            live |= slider_attacks(pt, s, pos.occ) & detonation_points;
        }
    }

    // Unblockable knight sweeps already crossing the king square.
    let knights = bits(pos.bb_of(them, PieceType::Knight))
        .filter(|&s| t.knight_span[s] & bb(k) != 0)
        .count() as i32;

    (latent, live.count_ones() as i32, knights)
}


/// Enemy pieces standing on any square our knights can sweep through.
fn knight_span_count(pos: &Position, us: Color) -> i32 {
    let t = &*TABLES;
    let enemy = pos.occ_color[us.flip().index()];
    bits(pos.bb_of(us, PieceType::Knight))
        .map(|s| (t.knight_span[s] & enemy).count_ones() as i32)
        .sum()
}

/// Our pieces adjacent to the enemy king — detonators already in place.
fn attack_primer_count(pos: &Position, us: Color) -> i32 {
    let Some(k) = pos.king_sq(us.flip()) else {
        return 0;
    };
    (TABLES.ring[k] & pos.occ_color[us.index()]).count_ones() as i32
}

/// Enemy pieces caught by the single best detonation available to `us`, or 0 if
/// the best one nets fewer than two.
fn cluster_count(pos: &Position, us: Color) -> i32 {
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
        best as i32
    } else {
        0
    }
}


// ---------------------------------------------------------------------------
// Tuning interface
// ---------------------------------------------------------------------------
//
// The evaluation is linear in its weights, so it can be written as a dot product
// of a per-position feature vector with the weight vector. That makes the spec §6
// weights — which are explicitly "guesses for a first alpha-beta pass, tune by
// self-play afterwards" — fittable against labelled self-play positions.
//
// Fitting these ten numbers costs nothing at run time: same code, different
// constants, identical speed. Every Elo found here is kept, unlike an NNUE where
// better judgement is paid for in search depth.

// Feature layout. Indices are stable; append rather than insert.
pub const F_MATERIAL: usize = 0; // 5, scaled by the endgame factor
pub const F_LATENT: usize = 5;
pub const F_LIVE: usize = 6;
pub const F_KNIGHT_ON_KING: usize = 7;
pub const F_CLUSTER: usize = 8;
pub const F_TEMPO: usize = 9;
/// Piece-square tables: 6 piece types x 10 symmetry orbits.
pub const F_PST: usize = 10;
/// Enemy pieces standing on our knights' sweep paths — the offensive counterpart
/// to `knight_on_king`. Spec §5.4 calls the knight the premium piece, but the
/// evaluation only modelled it defensively.
pub const F_KNIGHT_SPAN: usize = 70;
/// Our own pieces adjacent to the *enemy* king: detonators we have already
/// manufactured (spec §5.2).
pub const F_ATTACK_PRIMER: usize = 71;
/// Extra pawn value as the board empties, since the kings-and-pawns draw (§5.5)
/// makes pawns the trailing side's resource. A single linear pawn term cannot
/// express this.
pub const F_PAWN_ENDGAME: usize = 72;

pub const NUM_EVAL_FEATURES: usize = 73;

/// Number of distinct squares under the board's dihedral symmetry.
///
/// The rules are fully invariant under the dihedral group (no castling, no
/// forward direction), so a piece-square table **must** be too — a1, a8, h1 and
/// h8 cannot have different values. That collapses 64 squares to 10 orbits, so a
/// full six-piece table is 60 parameters rather than 384. Fewer numbers to fit,
/// impossible to overfit, and correct by construction.
pub const NUM_ORBITS: usize = 10;

/// `ORBIT[square]` in `0..10`.
pub static ORBIT: std::sync::LazyLock<[u8; 64]> = std::sync::LazyLock::new(|| {
    // Every orbit is identified by the unordered pair of distances-to-nearest-edge
    // along each axis. There are 10 such pairs from {0,1,2,3}.
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for a in 0..4 {
        for b in a..4 {
            pairs.push((a, b));
        }
    }
    debug_assert_eq!(pairs.len(), NUM_ORBITS);
    let mut table = [0u8; 64];
    for (s, slot) in table.iter_mut().enumerate() {
        let (r, c) = (rank_of(s), file_of(s));
        let a = r.min(7 - r);
        let b = c.min(7 - c);
        let key = (a.min(b), a.max(b));
        *slot = pairs.iter().position(|&p| p == key).unwrap() as u8;
    }
    table
});

pub fn eval_feature_name(i: usize) -> String {
    const BASE: [&str; 10] = [
        "pawn",
        "knight",
        "bishop",
        "rook",
        "queen",
        "latent_detonator",
        "live_detonator",
        "knight_on_king",
        "cluster",
        "tempo",
    ];
    match i {
        0..=9 => BASE[i].to_string(),
        _ if i < F_KNIGHT_SPAN => {
            let k = i - F_PST;
            format!(
                "pst_{}_orbit{}",
                PieceType::from_index(k / NUM_ORBITS).to_char(),
                k % NUM_ORBITS
            )
        }
        F_KNIGHT_SPAN => "knight_span".to_string(),
        F_ATTACK_PRIMER => "attack_primer".to_string(),
        F_PAWN_ENDGAME => "pawn_endgame".to_string(),
        _ => format!("feature{i}"),
    }
}

/// The weights the engine currently plays with, in feature order. Re-running the
/// tuner starts from these, so successive passes compound rather than restart.
pub fn eval_weights() -> [f32; NUM_EVAL_FEATURES] {
    TUNED_WEIGHTS.map(|w| w as f32)
}

/// Feature vector from **White's** point of view, such that
/// `dot(eval_features(pos), eval_weights()) ~= evaluate_white(pos) + tempo`.
///
/// The material features carry the endgame scale factor, which itself depends on
/// the piece values being fitted. It is frozen here at the currently compiled-in
/// weights, which is what makes the model linear and fittable; re-running the
/// tuner after adopting a new set picks up the second-order correction.
pub fn eval_features(pos: &Position) -> [f32; NUM_EVAL_FEATURES] {
    eval_features_with(pos, &TUNED_WEIGHTS)
}

pub fn eval_features_with(pos: &Position, scale_weights: &HandWeights) -> [f32; NUM_EVAL_FEATURES] {
    let scale = endgame_scale(pos, scale_weights) as f32 / 100.0;

    let mut f = [0f32; NUM_EVAL_FEATURES];
    for (i, pt) in [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
    ]
    .into_iter()
    .enumerate()
    {
        let diff = pos.bb_of(Color::White, pt).count_ones() as f32
            - pos.bb_of(Color::Black, pt).count_ones() as f32;
        f[i] = diff * scale;
    }

    let (lw, vw, kw) = king_safety_counts(pos, Color::White);
    let (lb, vb, kb) = king_safety_counts(pos, Color::Black);
    f[F_LATENT] = (lw - lb) as f32;
    f[F_LIVE] = (vw - vb) as f32;
    f[F_KNIGHT_ON_KING] = (kw - kb) as f32;
    f[F_CLUSTER] = (cluster_count(pos, Color::White) - cluster_count(pos, Color::Black)) as f32;
    f[F_TEMPO] = if pos.side == Color::White { 1.0 } else { -1.0 };

    let orbit = &*ORBIT;
    for p in 0..NUM_PIECES {
        let board = pos.pieces[p];
        if board == 0 {
            continue;
        }
        let base = F_PST + piece_type_of(p).index() * NUM_ORBITS;
        let sign = if piece_color(p) == Color::White { 1.0 } else { -1.0 };
        for s in bits(board) {
            f[base + orbit[s] as usize] += sign;
        }
    }

    f[F_KNIGHT_SPAN] =
        (knight_span_count(pos, Color::White) - knight_span_count(pos, Color::Black)) as f32;
    f[F_ATTACK_PRIMER] =
        (attack_primer_count(pos, Color::White) - attack_primer_count(pos, Color::Black)) as f32;
    f[F_PAWN_ENDGAME] = pawn_endgame_term(pos, endgame_scale(pos, scale_weights)) as f32;
    f
}

/// Evaluate from White's point of view using an arbitrary weight vector.
#[inline]
pub fn eval_with_weights(features: &[f32; NUM_EVAL_FEATURES], w: &[f32; NUM_EVAL_FEATURES]) -> f32 {
    features.iter().zip(w.iter()).map(|(f, w)| f * w).sum()
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
    fn a_reachable_detonation_square_next_to_the_king_is_punished() {
        // Spec §5.1, in the form that survived tuning: what matters is whether an
        // enemy slider can actually *land* on an occupied square of the king's
        // 3x3. Mere crowding turned out to be a mild asset (see the note on
        // `king_safety_counts`), so this tests the live term, not the latent one.
        //
        // Black king on e8 with a pawn on e7. The white rook on e1 has a clear
        // file to e7, so landing there detonates and takes the king with it.
        let exposed = Position::from_fen("4k3/4p3/8/p7/8/8/8/K3R3 w - - 0 1").unwrap();
        let (_, live, _) = king_safety_counts(&exposed, Color::Black);
        assert!(live > 0, "e7 is a reachable detonation square");

        // The same material, but the spare black pawn now blocks the file on e4.
        // Moving the blocker rather than adding one keeps material identical, so
        // the evaluation difference below is the threat and nothing else.
        let shielded = Position::from_fen("4k3/4p3/8/8/4p3/8/8/K3R3 w - - 0 1").unwrap();
        let (_, live_blocked, _) = king_safety_counts(&shielded, Color::Black);
        assert_eq!(live_blocked, 0, "the blocked rook cannot reach e7");
        assert_eq!(
            exposed.material_cp(Color::Black),
            shielded.material_cp(Color::Black),
            "the two positions must differ only in the threat"
        );

        assert!(
            evaluate_white(&exposed) > evaluate_white(&shielded),
            "a live detonation threat against the black king must favour white"
        );
    }

    #[test]
    fn tuned_weights_are_what_the_engine_plays() {
        // The gauntlet result (+49 Elo) was measured with the tuned set; make sure
        // a later edit cannot quietly revert the default to the spec guesses.
        let pos = Position::from_fen("4k3/8/8/3n4/8/8/8/R3K3 w - - 0 1").unwrap();
        assert_eq!(evaluate(&pos), evaluate_with(&pos, &TUNED_WEIGHTS));
        assert_ne!(SPEC_WEIGHTS, TUNED_WEIGHTS);
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
    fn feature_model_reproduces_the_evaluation() {
        // The tuner fits `eval_features . eval_weights`. If that drifts from what
        // `evaluate` actually computes, the tuned weights optimise the wrong
        // function and the whole exercise is silently worthless.
        //
        // Tolerance is a few centipawns: the real evaluation truncates to integers
        // in two places (the endgame scale, and the scaled material total) where
        // the feature model stays in floating point.
        use crate::movegen::{generate_vec, GenMode};
        use crate::tables::SplitMix64;

        let w = eval_weights();
        let mut rng = SplitMix64::new(20260802);
        let mut checked = 0;
        for _ in 0..400 {
            let mut pos = Position::startpos();
            for _ in 0..rng.below(70) {
                if pos.result().is_some() {
                    break;
                }
                let moves = generate_vec(&pos, GenMode::All);
                pos.make_move(moves[rng.below(moves.len())]);
            }
            if pos.result().is_some() {
                continue;
            }
            let modelled = eval_with_weights(&eval_features(&pos), &w);
            // `evaluate_white` excludes tempo; the feature model includes it.
            let tempo = TUNED_WEIGHTS[9] as f32;
            let actual = evaluate_white(&pos) as f32
                + if pos.side == Color::White { tempo } else { -tempo };
            assert!(
                (modelled - actual).abs() <= 3.0,
                "feature model {modelled} vs evaluation {actual} on {}",
                pos.to_fen()
            );
            checked += 1;
        }
        assert!(checked > 100, "not enough positions exercised");
    }

    #[test]
    fn king_capture_dominates_ordering() {
        let pos = Position::from_fen("8/8/8/8/8/8/4k3/4K3 w - - 0 1").unwrap();
        let mv = Move::parse("e1e2").unwrap();
        assert_eq!(move_delta(&pos, mv), KING_ORDER_VALUE);
    }
}
