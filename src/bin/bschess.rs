//! Mode A — operator mode (build plan, Phase 3).
//!
//! You are the interface between this engine and your opponent's. You type their
//! move, the engine replies with its own.
//!
//! After every move — yours or the engine's — this prints the resulting board,
//! **the list of squares emptied by the move**, the engine's evaluation, depth and
//! principal variation, and the current ply number.
//!
//! The emptied-squares line is not a nicety. A blast clears up to nine squares; if
//! the two engines disagree about any rule the boards diverge silently and neither
//! player notices for many moves (spec §8). Printing what was destroyed lets your
//! opponent compare against their own output immediately.

use bschess::eval::{evaluate, Evaluator};
use bschess::movegen::{generate_vec, GenMode};
use bschess::nnue::Network;
use bschess::position::{MoveInfo, Position};
use bschess::search::{SearchLimits, SearchOptions, Searcher};
use bschess::types::*;
use std::io::{BufRead, Write};
use std::sync::Arc;

struct Session {
    pos: Position,
    /// Every position from the start of the game, so `undo` is exact.
    history: Vec<Position>,
    played: Vec<Move>,
    engine_color: Option<Color>,
    searcher: Searcher,
    limits: SearchLimits,
}

impl Session {
    fn new() -> Session {
        Session {
            pos: Position::startpos(),
            history: Vec::new(),
            played: Vec::new(),
            engine_color: None,
            searcher: Searcher::new(SearchOptions {
                verbose: false,
                ..Default::default()
            }),
            limits: SearchLimits::movetime_ms(2000),
        }
    }

    fn reset(&mut self, engine_color: Option<Color>) {
        self.pos = Position::startpos();
        self.history.clear();
        self.played.clear();
        self.engine_color = engine_color;
        self.searcher.reset();
    }

    fn apply(&mut self, mv: Move, who: &str) -> MoveInfo {
        let before = self.pos;
        self.history.push(before);
        self.played.push(mv);
        let info = self.pos.make_move(mv);
        print_move_report(&before, mv, &info, who);
        print_state(&self.pos);
        info
    }

    fn undo(&mut self) -> bool {
        match self.history.pop() {
            Some(p) => {
                self.pos = p;
                self.played.pop();
                true
            }
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn piece_letter(pos: &Position, s: usize) -> String {
    match pos.piece_at(s) {
        None => "?".to_string(),
        Some(p) => {
            let c = piece_type_of(p).to_char();
            let c = if piece_color(p) == Color::White {
                c.to_ascii_uppercase()
            } else {
                c
            };
            format!("{c}{}", square_name(s))
        }
    }
}

fn print_move_report(before: &Position, mv: Move, info: &MoveInfo, who: &str) {
    let pt = piece_type_of(info.piece);
    let kind = match pt {
        PieceType::Pawn => "teleport".to_string(),
        PieceType::Knight => {
            if info.captured != 0 {
                format!("knight sweep, {} taken", info.captured.count_ones())
            } else {
                "knight sweep".to_string()
            }
        }
        PieceType::Rook | PieceType::Bishop => {
            if info.mover_died {
                "DETONATION".to_string()
            } else {
                "quiet slide".to_string()
            }
        }
        PieceType::Queen => {
            if info.captured != 0 {
                format!("queen sweep, {} taken", info.captured.count_ones())
            } else {
                "queen slide".to_string()
            }
        }
        PieceType::King => {
            if info.captured != 0 {
                "king capture".to_string()
            } else {
                "king step".to_string()
            }
        }
    };

    println!("\n{who} plays {mv}   [{kind}]");

    // The cross-check line. Identities are read from the *pre-move* board.
    let mut destroyed: Vec<String> = bits(info.captured)
        .map(|s| piece_letter(before, s))
        .collect();
    if info.mover_died {
        destroyed.push(format!("{} (mover)", piece_letter(before, mv.from())));
    }
    destroyed.sort();
    println!(
        "  destroyed: {}",
        if destroyed.is_empty() {
            "-".to_string()
        } else {
            destroyed.join(" ")
        }
    );
    println!(
        "  emptied:   {}",
        if info.emptied == 0 {
            "-".to_string()
        } else {
            square_list(info.emptied)
        }
    );
}

fn print_state(pos: &Position) {
    print!("{}", pos.board_string());
    println!("  ply {}   {} to move", pos.ply, pos.side.name());
    println!("  fen {}", pos.to_fen());
    if let Some(r) = pos.result() {
        println!("\n  *** GAME OVER — {} ***", r.describe());
    }
}

fn print_search(r: &bschess::search::SearchResult) {
    println!(
        "  eval {}  depth {}/{}  nodes {}  nps {}  time {}ms",
        r.score_string(),
        r.depth,
        r.seldepth,
        r.nodes,
        r.nps(),
        r.elapsed.as_millis()
    );
    println!("  pv {}", r.pv_string());
}

fn help() {
    println!(
        "\
commands:
  new [w|b]      start a game; the letter is the colour the ENGINE plays
  <move>         e.g. e2e4 — apply that move, then search and reply
  go             force the engine to move now
  board          reprint the board
  fen            print the current position as FEN
  setfen <fen>   load a position (use this to resync with your opponent)
  undo           take back the last ply
  moves          list legal moves from the current position
  time <ms>      thinking time per move   (current default 2000)
  depth <n>      search to a fixed depth instead of a time limit
  eval           static evaluation of the current position
  net <path>     load NNUE weights; `net off` reverts to the hand evaluation
  perft <n>      node counts to depth n, with a per-move breakdown
  help / quit"
    );
}

// ---------------------------------------------------------------------------

/// Non-interactive: read FENs on stdin, print `fen<TAB>eval_cp` for each.
///
/// This exists so `tools/verify_net.py` can check that the exported weights
/// evaluate identically here and in PyTorch. A transposed weight matrix would
/// otherwise be completely silent — the engine would just play badly.
fn eval_fens(net_path: Option<&str>) -> i32 {
    let evaluator = match net_path {
        Some(p) => match Network::load(p) {
            Ok(n) => Evaluator::Nnue(Arc::new(n)),
            Err(e) => {
                eprintln!("could not load {p}: {e}");
                return 2;
            }
        },
        None => Evaluator::Hand,
    };
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let fen = line.trim_start_matches('\u{feff}').trim();
        if fen.is_empty() || fen.starts_with('#') {
            continue;
        }
        match Position::from_fen(fen) {
            Ok(p) => println!("{fen}\t{}", evaluator.eval(&p)),
            Err(e) => {
                eprintln!("bad FEN '{fen}': {e}");
                return 2;
            }
        }
    }
    0
}

/// Time the NNUE forward pass and its parts, so optimisation targets the layer
/// that actually costs something rather than the one that looks expensive.
fn bench_nnue(net_path: &str) {
    use bschess::nnue::{L1, L2, MAX_HL};
    let net = match Network::load(net_path) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("could not load {net_path}: {e}");
            std::process::exit(2);
        }
    };
    // A spread of realistic positions rather than one, so the feature-column
    // access pattern is as scattered as it is in a real search.
    let positions: Vec<Position> = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w - - 30 16",
        "4k3/8/8/3n4/8/8/8/R3K3 w - - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 44 23",
        "3qk3/8/8/8/8/8/8/3QK3 b - - 3 2",
    ]
    .iter()
    .map(|f| Position::from_fen(f).unwrap())
    .collect();

    const ITERS: usize = 200_000;
    let mut sink = 0i64;

    let t = std::time::Instant::now();
    for i in 0..ITERS {
        sink += net.evaluate(&positions[i % positions.len()]) as i64;
    }
    let full = t.elapsed().as_secs_f64();

    let mut acc = [[0.0f32; MAX_HL]; 2];
    let t = std::time::Instant::now();
    for i in 0..ITERS {
        net.accumulate(&positions[i % positions.len()], &mut acc);
        sink += acc[0][0] as i64;
    }
    let accum = t.elapsed().as_secs_f64();

    // How many L1 inputs are exactly zero? They come out of a ClippedReLU, so
    // everything the accumulator drove negative clamps to zero, and a zero input
    // contributes nothing to any output. If that fraction were large, a
    // column-major weight layout could skip those columns and save the memory
    // traffic, not just the arithmetic. Measured at ~7% here, so it cannot.
    let mut zero = 0usize;
    let mut total = 0usize;
    for pos in &positions {
        net.accumulate(pos, &mut acc);
        for half in acc.iter() {
            for v in half[..net.hl].iter() {
                if v.clamp(0.0, 1.0) == 0.0 {
                    zero += 1;
                }
                total += 1;
            }
        }
    }

    let per = |s: f64| s / ITERS as f64 * 1e9;
    println!("network            768x{}x{}x{}, {} parameters", net.hl, L1, L2, net.parameter_count());
    println!("evaluate           {:>8.0} ns   {:>10.0} evals/s", per(full), ITERS as f64 / full);
    println!("  accumulator      {:>8.0} ns   {:>5.1}% of the total", per(accum), 100.0 * accum / full);
    println!("  layers L1-L3     {:>8.0} ns   {:>5.1}% of the total", per(full - accum), 100.0 * (full - accum) / full);
    println!(
        "L1 input sparsity  {:>8.1}%  of {total} values clamp to exactly zero",
        100.0 * zero as f64 / total as f64
    );
    println!("(checksum {sink})");
}

fn main() {
    bschess::init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(i) = args.iter().position(|a| a == "--eval-fens") {
        std::process::exit(eval_fens(args.get(i + 1).map(|s| s.as_str())));
    }
    if let Some(i) = args.iter().position(|a| a == "--bench-nnue") {
        bench_nnue(args.get(i + 1).map(|s| s.as_str()).unwrap_or("nets/gen1.bin"));
        return;
    }

    let mut s = Session::new();

    println!("Bullshit Chess engine — operator mode.");
    println!("Type `help` for commands. Moves are long algebraic: e2e4, a1h8.\n");
    print_state(&s.pos);

    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        print!("> ");
        let _ = std::io::stdout().flush();
        let Some(Ok(line)) = lines.next() else { break };
        // Strip a UTF-8 BOM: piping a script into the engine on Windows prefixes
        // one to the first line, which would otherwise look like a bad command.
        let line = line.trim_start_matches('\u{feff}').trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let cmd = parts.next().unwrap();
        let rest: Vec<&str> = parts.collect();

        match cmd {
            "quit" | "exit" | "q" => break,
            "help" | "?" => help(),

            "new" => {
                let color = match rest.first().map(|s| s.to_ascii_lowercase()) {
                    Some(c) if c.starts_with('w') => Some(Color::White),
                    Some(c) if c.starts_with('b') => Some(Color::Black),
                    None => None,
                    Some(other) => {
                        println!("  ?? unknown colour '{other}' — use w or b");
                        continue;
                    }
                };
                s.reset(color);
                match color {
                    Some(c) => println!("  new game, engine plays {}", c.name()),
                    None => println!("  new game, engine plays neither side (use `go`)"),
                }
                print_state(&s.pos);
                if color == Some(Color::White) {
                    engine_move(&mut s);
                }
            }

            "board" => print_state(&s.pos),
            "fen" => println!("{}", s.pos.to_fen()),

            "setfen" => {
                if rest.is_empty() {
                    println!("  ?? usage: setfen <fen>");
                    continue;
                }
                match Position::from_fen(&rest.join(" ")) {
                    Ok(p) => {
                        s.pos = p;
                        s.history.clear();
                        s.played.clear();
                        s.searcher.reset();
                        println!("  position loaded");
                        print_state(&s.pos);
                    }
                    Err(e) => println!("  ?? bad FEN: {e}"),
                }
            }

            "undo" => {
                if s.undo() {
                    println!("  took back one ply");
                    print_state(&s.pos);
                } else {
                    println!("  ?? nothing to undo");
                }
            }

            "moves" => {
                let all = generate_vec(&s.pos, GenMode::All);
                let searched = generate_vec(&s.pos, GenMode::Search);
                let mut names: Vec<String> = all.iter().map(|m| m.to_string()).collect();
                names.sort();
                println!("  {} legal moves:", names.len());
                for chunk in names.chunks(16) {
                    println!("    {}", chunk.join(" "));
                }
                println!(
                    "  ({} of them are searched; the rest are pawn teleports the \
                     restricted generator prunes)",
                    searched.len()
                );
            }

            "time" => match rest.first().and_then(|v| v.parse::<u64>().ok()) {
                Some(ms) => {
                    s.limits = SearchLimits::movetime_ms(ms);
                    println!("  thinking time set to {ms} ms per move");
                }
                None => println!("  ?? usage: time <milliseconds>"),
            },

            "depth" => match rest.first().and_then(|v| v.parse::<i32>().ok()) {
                Some(d) => {
                    s.limits = SearchLimits::depth(d);
                    println!("  searching to fixed depth {d}");
                }
                None => println!("  ?? usage: depth <n>"),
            },

            "eval" => {
                let hand = evaluate(&s.pos);
                println!(
                    "  hand eval {:+.2} (side to move: {})",
                    hand as f64 / 100.0,
                    s.pos.side.name()
                );
                if let Evaluator::Nnue(n) = &s.searcher.options.evaluator {
                    println!("  nnue eval {:+.2}", n.evaluate(&s.pos) as f64 / 100.0);
                }
            }

            "net" => match rest.first() {
                Some(&"off") => {
                    s.searcher.options.evaluator = Evaluator::Hand;
                    println!("  reverted to the hand evaluation");
                }
                Some(path) => match Network::load(path) {
                    Ok(n) => {
                        println!("  loaded {} parameters from {path}", n.parameter_count());
                        s.searcher.options.evaluator = Evaluator::Nnue(Arc::new(n));
                    }
                    Err(e) => println!("  ?? could not load {path}: {e}"),
                },
                None => println!("  current evaluator: {}", s.searcher.options.evaluator.name()),
            },

            "perft" => match rest.first().and_then(|v| v.parse::<u32>().ok()) {
                Some(d) => {
                    let t = std::time::Instant::now();
                    let divide = bschess::perft::perft_divide(&s.pos, d);
                    let total: u64 = divide.iter().map(|(_, n)| n).sum();
                    if d <= 2 {
                        for (m, n) in &divide {
                            println!("  {m}: {n}");
                        }
                    }
                    println!("  perft({d}) = {total}   [{} ms]", t.elapsed().as_millis());
                }
                None => println!("  ?? usage: perft <depth>"),
            },

            "go" => engine_move(&mut s),

            _ => {
                let Some(mv) = Move::parse(cmd) else {
                    println!("  ?? unknown command '{cmd}' — type `help`");
                    continue;
                };
                if s.pos.result().is_some() {
                    println!("  ?? the game is already over — `new` to start again");
                    continue;
                }
                let legal = generate_vec(&s.pos, GenMode::All);
                if !legal.contains(&mv) {
                    println!("  ?? {mv} is not legal here. `moves` lists what is.");
                    continue;
                }
                let who = s.pos.side.name().to_string();
                s.apply(mv, &who);
                if s.pos.result().is_none() && Some(s.pos.side) == s.engine_color {
                    engine_move(&mut s);
                }
            }
        }
    }
}

fn engine_move(s: &mut Session) {
    if s.pos.result().is_some() {
        println!("  ?? the game is over");
        return;
    }
    let limits = s.limits.clone();
    let r = s.searcher.search(&s.pos, &limits);
    if r.best.is_none() {
        println!("  ?? the engine found no move — this is a bug (spec §1.4)");
        return;
    }
    let who = format!("engine ({})", s.pos.side.name());
    s.apply(r.best, &who);
    print_search(&r);
}
