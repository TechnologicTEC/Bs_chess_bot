//! The spec §7 test suite, checkbox by checkbox.
//!
//! Hand-verified before any training code exists. Rule bugs here would train a
//! confident engine that plays a subtly different game from the opponent's.

use bschess::movegen::{generate_vec, GenMode};
use bschess::position::{GameResult, Position};
use bschess::types::*;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn pos(fen: &str) -> Position {
    Position::from_fen(fen).unwrap_or_else(|e| panic!("bad test FEN {fen}: {e}"))
}

fn mv(s: &str) -> Move {
    Move::parse(s).unwrap_or_else(|| panic!("bad test move {s}"))
}

/// Sorted, space-separated square names, for readable assertions.
fn squares(b: Bitboard) -> String {
    square_list(b)
}

fn assert_legal(p: &Position, m: Move) {
    assert!(
        generate_vec(p, GenMode::All).contains(&m),
        "{m} should be legal in {}",
        p.to_fen()
    );
}

fn assert_illegal(p: &Position, m: Move) {
    assert!(
        !generate_vec(p, GenMode::All).contains(&m),
        "{m} should NOT be legal in {}",
        p.to_fen()
    );
}

/// Destinations of every generated move from `from`, sorted.
fn destinations(p: &Position, from: &str) -> Vec<String> {
    let f = parse_square(from).unwrap();
    let mut v: Vec<String> = generate_vec(p, GenMode::All)
        .into_iter()
        .filter(|m| m.from() == f)
        .map(|m| square_name(m.to()))
        .collect();
    v.sort();
    v
}

// ---------------------------------------------------------------------------
// Explosions — spec §2.3
// ---------------------------------------------------------------------------

#[test]
fn rook_capture_on_a_corner_clears_the_clipped_2x2_and_removes_the_mover() {
    // Rb1 takes na1. Blast is the 3x3 centred on a1, clipped to {a1,a2,b1,b2}.
    let p = pos("4k2r/8/8/8/8/8/Np6/nR2K3 w - - 0 1");
    let (after, info) = p.after(mv("b1a1"));

    assert_eq!(squares(info.emptied), "a1 a2 b1 b2");
    assert!(info.mover_died, "the rook does not survive its own blast");
    assert_eq!(
        after.occ.count_ones(),
        3,
        "only the two kings and the black rook should remain: {}",
        after.to_fen()
    );
    for s in ["a1", "a2", "b1", "b2"] {
        assert!(after.piece_at(parse_square(s).unwrap()).is_none(), "{s} not cleared");
    }
    after.validate().unwrap();
}

#[test]
fn rook_capture_destroys_the_movers_own_queen_next_to_the_target() {
    // Rd1 takes pd4. Wc4 stands adjacent to the target and goes up with it.
    let p = pos("4k2r/8/8/8/2Qp4/8/8/3RK3 w - - 0 1");
    assert_eq!(p.piece_at(parse_square("c4").unwrap()), Some(piece_index(Color::White, PieceType::Queen)));

    let (after, info) = p.after(mv("d1d4"));
    assert!(info.mover_died);
    assert_eq!(squares(info.emptied), "c4 d1 d4");
    assert_eq!(
        after.bb_of(Color::White, PieceType::Queen),
        0,
        "white's own queen should have been destroyed"
    );
    assert_eq!(after.bb_of(Color::White, PieceType::Rook), 0);
    after.validate().unwrap();
}

#[test]
fn blast_containing_both_kings_is_a_draw_not_a_win() {
    // Ba1 takes pd4; the 3x3 around d4 contains Ke4 and kd5.
    let p = pos("8/8/8/3k4/3pK3/8/8/B7 w - - 0 1");
    let (after, _) = p.after(mv("a1d4"));
    assert_eq!(after.king_sq(Color::White), None);
    assert_eq!(after.king_sq(Color::Black), None);
    assert_eq!(after.result(), Some(GameResult::Draw));
}

#[test]
fn blast_containing_only_the_movers_own_king_loses_for_the_mover() {
    // Spec §3.2: destroying your own king is a loss for you regardless of who moved.
    let p = pos("7k/8/8/4K3/3p4/8/8/B7 w - - 0 1");
    let (after, _) = p.after(mv("a1d4"));
    assert_eq!(after.king_sq(Color::White), None);
    assert!(after.king_sq(Color::Black).is_some());
    assert_eq!(after.result(), Some(GameResult::Win(Color::Black)));
}

#[test]
fn rook_cannot_slide_past_a_blocking_pawn_to_a_target_behind_it() {
    // Ra1 with a black pawn on a4 and a black rook on a7 behind it.
    let p = pos("4k3/r7/8/8/p7/8/8/R3K3 w - - 0 1");
    let north: Vec<String> = destinations(&p, "a1")
        .into_iter()
        .filter(|s| s.starts_with('a'))
        .collect();
    assert_eq!(north, vec!["a2", "a3", "a4"]);
    assert_illegal(&p, mv("a1a7"));
}

#[test]
fn a_blast_does_not_chain_react() {
    // Spec §2.3: a rook killed by someone else's blast does not detonate in turn.
    // Rd1 takes pd4; the black rook on e5 is inside the blast and simply dies.
    let p = pos("4k3/8/8/4r3/3p4/8/8/3RK2q w - - 0 1");
    let (after, info) = p.after(mv("d1d4"));
    assert!(info.captured & bb(parse_square("e5").unwrap()) != 0);
    // Only the 3x3 around d4 plus the mover's origin are affected — nothing else.
    assert_eq!(squares(info.emptied), "d1 d4 e5");
    after.validate().unwrap();
}

// ---------------------------------------------------------------------------
// Knight sweeps — spec §2.2
// ---------------------------------------------------------------------------

#[test]
fn knight_sweep_removes_two_friendly_pieces_and_survives_on_the_destination() {
    // Nd4 plays (+2,+1): path d5, d6, e6. Both intermediates are friendly pawns.
    let p = pos("4k3/8/3P4/3P4/3N4/8/8/4K3 w - - 0 1");
    let (after, info) = p.after(mv("d4e6"));

    assert_eq!(squares(info.captured), "d5 d6");
    assert!(!info.mover_died, "the knight survives its own sweep");
    assert_eq!(
        after.piece_at(parse_square("e6").unwrap()),
        Some(piece_index(Color::White, PieceType::Knight))
    );
    assert!(after.piece_at(parse_square("d4").unwrap()).is_none());
    assert_eq!(after.bb_of(Color::White, PieceType::Pawn), 0);
    after.validate().unwrap();
}

#[test]
fn knight_capturing_the_king_by_passing_over_it_wins() {
    // The black king sits on d6, an intermediate square. The knight lands on an
    // empty e6 — it never touches the king's square, and still wins.
    let p = pos("8/8/3k4/8/3N4/8/8/4K3 w - - 0 1");
    assert!(p.piece_at(parse_square("e6").unwrap()).is_none());
    let (after, info) = p.after(mv("d4e6"));
    assert_eq!(squares(info.captured), "d6");
    assert_eq!(after.result(), Some(GameResult::Win(Color::White)));
}

#[test]
fn a_fully_occupied_knight_path_is_still_legal() {
    // Spec §2.2: the knight cannot be blocked. There is no such thing as an
    // illegal knight move due to occupancy.
    let p = pos("4k3/8/3ppp2/3p4/3N4/8/8/4K3 w - - 0 1");
    assert_legal(&p, mv("d4e6"));
    let (after, info) = p.after(mv("d4e6"));
    assert_eq!(squares(info.captured), "d5 d6 e6");
    assert_eq!(
        after.piece_at(parse_square("e6").unwrap()),
        Some(piece_index(Color::White, PieceType::Knight))
    );
    after.validate().unwrap();
}

#[test]
fn every_knight_destination_on_the_board_is_generated() {
    // From d4 all eight L-shapes are on the board, occupancy notwithstanding.
    let p = pos("4k3/8/2ppp3/2ppp3/2pNp3/2ppp3/8/4K3 w - - 0 1");
    assert_eq!(destinations(&p, "d4").len(), 8);
}

// ---------------------------------------------------------------------------
// Queen sweeps — spec §2.4
// ---------------------------------------------------------------------------

#[test]
fn queen_ray_with_blockers_at_distance_3_and_6_reaches_exactly_1_through_6() {
    // Spec §2.4's worked example, verbatim.
    let p = pos("8/1p6/8/8/1p6/8/8/1Q2k2K w - - 0 1");
    let north: Vec<String> = destinations(&p, "b1")
        .into_iter()
        .filter(|s| s.starts_with('b'))
        .collect();
    assert_eq!(north, vec!["b2", "b3", "b4", "b5", "b6", "b7"]);
}

#[test]
fn queen_may_not_reach_distance_7_past_two_pieces() {
    let p = pos("8/1p6/8/8/1p6/8/8/1Q2k2K w - - 0 1");
    assert_illegal(&p, mv("b1b8"));
}

#[test]
fn queen_capturing_two_of_her_own_pieces_is_legal() {
    let p = pos("4k3/1P6/8/8/1P6/8/8/1Q5K w - - 0 1");
    assert_legal(&p, mv("b1b7"));
    let (after, info) = p.after(mv("b1b7"));
    assert_eq!(squares(info.captured), "b4 b7");
    assert_eq!(after.bb_of(Color::White, PieceType::Pawn), 0);
    assert_eq!(
        after.piece_at(parse_square("b7").unwrap()),
        Some(piece_index(Color::White, PieceType::Queen)),
        "the queen survives her sweep"
    );
    after.validate().unwrap();
}

#[test]
fn queen_captures_one_enemy_piece_and_stops_on_the_next_empty_square() {
    let p = pos("4k3/8/8/8/1p6/8/8/1Q5K w - - 0 1");
    let (after, info) = p.after(mv("b1b5"));
    assert_eq!(squares(info.captured), "b4");
    assert_eq!(
        after.piece_at(parse_square("b5").unwrap()),
        Some(piece_index(Color::White, PieceType::Queen))
    );
    after.validate().unwrap();
}

#[test]
fn queen_stops_no_later_than_her_second_capture() {
    // Three pieces in a row: she may take the first two and must stop there.
    let p = pos("4k3/8/8/8/8/1p6/1p6/1Q4pK w - - 0 1");
    let north: Vec<String> = destinations(&p, "b1")
        .into_iter()
        .filter(|s| s.starts_with('b'))
        .collect();
    assert_eq!(north, vec!["b2", "b3"]);
}

// ---------------------------------------------------------------------------
// Pawns — spec §2.1
// ---------------------------------------------------------------------------

#[test]
fn pawn_cannot_move_to_an_occupied_square_even_an_undefended_enemy_piece() {
    let p = pos("4k3/8/8/8/3n4/8/3P4/4K3 w - - 0 1");
    assert_illegal(&p, mv("d2d4"));
    // ...but any empty square is fine, however far away.
    assert_legal(&p, mv("d2h7"));
}

#[test]
fn pawn_on_the_eighth_rank_does_not_promote_and_may_still_teleport() {
    let p = pos("P3k3/8/8/8/8/8/8/4K1R1 w - - 0 1");
    let a8 = parse_square("a8").unwrap();
    assert_eq!(
        p.piece_at(a8),
        Some(piece_index(Color::White, PieceType::Pawn))
    );
    assert_legal(&p, mv("a8d4"));
    let (after, _) = p.after(mv("a8d4"));
    assert_eq!(
        after.piece_at(parse_square("d4").unwrap()),
        Some(piece_index(Color::White, PieceType::Pawn)),
        "it is still a pawn"
    );
    assert_eq!(after.bb_of(Color::White, PieceType::Queen), 0);
}

#[test]
fn pawn_blocks_a_rook_ray() {
    // Spec §2.1: pawns block sliding pieces normally.
    let p = pos("4k3/8/8/8/8/8/P7/R3K3 w - - 0 1");
    let north: Vec<String> = destinations(&p, "a1")
        .into_iter()
        .filter(|s| s.starts_with('a'))
        .collect();
    assert_eq!(north, vec!["a2"], "the rook may only detonate on its own pawn");
}

#[test]
fn pawn_teleport_count_equals_pawns_times_empty_squares() {
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
        "4k3/8/8/8/8/8/PPP5/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    ] {
        let p = pos(fen);
        let pawns = p.bb_of(p.side, PieceType::Pawn).count_ones() as usize;
        let empty = 64 - p.occ.count_ones() as usize;
        let generated = generate_vec(&p, GenMode::All)
            .into_iter()
            .filter(|m| {
                p.piece_at(m.from()).map(piece_type_of) == Some(PieceType::Pawn)
            })
            .count();
        assert_eq!(generated, pawns * empty, "in {fen}");
    }
}

// ---------------------------------------------------------------------------
// King — spec §2.5
// ---------------------------------------------------------------------------

#[test]
fn king_may_capture_a_friendly_piece_and_may_step_onto_an_attacked_square() {
    let p = pos("4k3/8/8/8/8/8/4P3/4K2r w - - 0 1");
    assert_legal(&p, mv("e1e2")); // takes its own pawn
    assert_legal(&p, mv("e1f1")); // steps next to the enemy rook
    let (after, info) = p.after(mv("e1e2"));
    assert_eq!(squares(info.captured), "e2");
    assert_eq!(after.bb_of(Color::White, PieceType::Pawn), 0);
}

// ---------------------------------------------------------------------------
// Terminal conditions — spec §3
// ---------------------------------------------------------------------------

#[test]
fn simultaneous_double_king_removal_is_a_draw_by_explosion() {
    let p = pos("8/8/8/3k4/3pK3/8/8/B7 w - - 0 1");
    let (after, _) = p.after(mv("a1d4"));
    assert_eq!(after.result(), Some(GameResult::Draw));
}

#[test]
fn simultaneous_double_king_removal_is_a_draw_by_knight_sweep() {
    // Nd4 path is d5, d6, e6 — both kings stand on it.
    let p = pos("8/8/3k4/3K4/3N4/8/8/8 w - - 0 1");
    let (after, info) = p.after(mv("d4e6"));
    assert_eq!(squares(info.captured), "d5 d6");
    assert_eq!(after.result(), Some(GameResult::Draw));
}

#[test]
fn simultaneous_double_king_removal_is_a_draw_by_queen_sweep() {
    // Qb1 sweeps north over kb3 and stops on Kb5 — two captures, both kings.
    let p = pos("8/8/8/1K6/8/1k6/8/1Q6 w - - 0 1");
    let (after, info) = p.after(mv("b1b5"));
    assert_eq!(squares(info.captured), "b3 b5");
    assert_eq!(after.result(), Some(GameResult::Draw));
}

#[test]
fn kings_and_pawns_only_is_an_immediate_draw() {
    assert_eq!(
        pos("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1").result(),
        Some(GameResult::Draw)
    );
}

#[test]
fn two_bare_kings_is_a_draw() {
    assert_eq!(pos("4k3/8/8/8/8/8/8/4K3 w - - 0 1").result(), Some(GameResult::Draw));
}

#[test]
fn one_surviving_king_wins_regardless_of_who_moved() {
    assert_eq!(
        pos("4k3/8/8/8/8/8/8/4R3 w - - 0 1").result(),
        Some(GameResult::Win(Color::Black))
    );
    assert_eq!(
        pos("4r3/8/8/8/8/8/8/4K3 w - - 0 1").result(),
        Some(GameResult::Win(Color::White))
    );
}

#[test]
fn ply_cap_adjudicates_by_material_rather_than_drawing() {
    // Spec §4: at the cap, score by material with a threshold of 2.0 pawns'
    // worth, so that shuffling to the cap is not a free draw for the loser.
    assert_eq!(
        pos("4k3/8/8/8/8/8/8/3QK3 w - - 200 101").result(),
        Some(GameResult::Win(Color::White)),
        "a queen up at the cap is a win"
    );
    assert_eq!(
        pos("3qk3/8/8/8/8/8/8/3QK3 w - - 200 101").result(),
        Some(GameResult::Draw),
        "level material at the cap is a draw"
    );
    assert_eq!(
        pos("4k1n1/8/8/8/8/8/6P1/4K1N1 w - - 200 101").result(),
        Some(GameResult::Draw),
        "a single pawn (0.25) is inside the 2.0 threshold"
    );
    assert_eq!(
        pos("4k1n1/8/8/8/8/8/8/2B1K1N1 w - - 200 101").result(),
        Some(GameResult::Win(Color::White)),
        "a bishop (4.0) clears the 2.0 threshold"
    );
}

#[test]
fn the_game_is_still_live_one_ply_before_the_cap() {
    assert_eq!(pos("4k3/8/8/8/8/8/8/3QK3 w - - 199 100").result(), None);
}

// ---------------------------------------------------------------------------
// Spec §1.4 — no stalemate
// ---------------------------------------------------------------------------

#[test]
fn a_side_to_move_always_has_a_legal_move() {
    // The king may capture friendly pieces, so occupancy never restricts it; the
    // only limit is the board edge, and no square has all 8 neighbours off-board.
    for fen in [
        "4k3/8/8/8/8/8/8/KQ6 w - - 0 1",
        "4k3/8/8/8/8/8/PPP5/KPP5 w - - 0 1",
        "4k3/8/8/8/8/8/8/K6Q w - - 0 1",
    ] {
        let p = pos(fen);
        assert!(!generate_vec(&p, GenMode::All).is_empty(), "{fen}");
        assert!(!generate_vec(&p, GenMode::Search).is_empty(), "{fen} (search mode)");
    }
}

// ---------------------------------------------------------------------------
// Perft — spec §7, frozen from ref/reference.py
// ---------------------------------------------------------------------------

/// Node counts produced by the independent Python reference implementation in
/// `ref/reference.py`. Regenerate with:
///
///   python ref/reference.py --suite tests/perft_suite.txt
///
/// These are the Phase 1 gate. If one of them changes, movegen changed.
#[test]
fn perft_matches_the_independent_reference_implementation() {
    use bschess::perft::perft;

    let cases: &[(&str, u32, u64)] = &[
        // start position, depths 1-4
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1", 1, 292),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1", 2, 84_165),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1", 3, 24_169_988),
        // terminal positions expand to nothing
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 1, 0),
        ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", 2, 0),
        // sparse positions, where the rule interactions actually live
        ("4k3/8/8/3n4/8/8/8/R3K3 w - - 0 1", 4, 44_542),
        ("k7/8/8/8/8/8/8/K1Q5 w - - 0 1", 4, 10_248),
        ("7k/8/8/8/8/8/8/K6R w - - 0 1", 4, 4_348),
        ("1n2k3/8/8/8/8/8/8/1N2K3 w - - 0 1", 4, 6_560),
        ("4k3/8/8/8/8/8/8/B3K2B w - - 0 1", 4, 14_120),
        ("8/8/8/8/8/8/n7/R3K2k w - - 0 1", 4, 6_137),
        ("3qk3/8/8/8/8/8/8/3QK3 w - - 0 1", 3, 11_790),
        ("4k3/8/8/8/8/8/4P3/R3K3 w - - 0 1", 3, 27_929),
        ("4k3/pp6/8/8/8/8/6PP/R3K2R w - - 0 1", 3, 1_984_054),
        ("4k3/8/8/8/8/8/8/4K2Q b - - 0 1", 4, 19_656),
        ("k6K/8/8/8/8/8/8/6qR w - - 0 1", 4, 85_691),
    ];

    for &(fen, depth, expected) in cases {
        assert_eq!(
            perft(&pos(fen), depth),
            expected,
            "perft({depth}) mismatch for {fen}"
        );
    }
}
