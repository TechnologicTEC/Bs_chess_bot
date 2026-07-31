//! Perft driver — the Phase 1 gate.
//!
//!   perft [depth] [--fen "<fen>"] [--divide]
//!
//! Emits `depth<TAB>nodes` lines so the output diffs directly against
//! `python ref/reference.py`.

use bschess::perft::{perft_divide, perft_parallel};
use bschess::position::Position;

fn main() {
    bschess::init();
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut depth = 4u32;
    let mut fen = String::new();
    let mut divide = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--fen" => {
                i += 1;
                fen = args.get(i).cloned().unwrap_or_default();
            }
            "--divide" => divide = true,
            other => {
                if let Ok(d) = other.parse::<u32>() {
                    depth = d;
                }
            }
        }
        i += 1;
    }

    let pos = if fen.is_empty() {
        Position::startpos()
    } else {
        match Position::from_fen(&fen) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("bad FEN: {e}");
                std::process::exit(2);
            }
        }
    };

    eprintln!("{}", pos.board_string());
    eprintln!("fen {}", pos.to_fen());

    if divide {
        for (m, n) in perft_divide(&pos, depth) {
            println!("{m}\t{n}");
        }
        return;
    }

    for d in 1..=depth {
        let t = std::time::Instant::now();
        let n = perft_parallel(&pos, d);
        let ms = t.elapsed().as_millis();
        println!("{d}\t{n}");
        eprintln!("  depth {d}: {n} nodes in {ms} ms");
    }
}
