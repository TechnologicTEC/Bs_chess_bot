//! Phase 6 — measuring progress.
//!
//! Gate a candidate against the previous generation:
//!   gauntlet --pairs 100 --depth 6 --a nets/gen2.bin --b nets/gen1.bin
//!
//! Round-robin over a whole pool (leave `--a` off to include the hand eval):
//!   gauntlet --pairs 50 --depth 6 --pool nets/gen1.bin nets/gen2.bin nets/gen3.bin
//!
//! Elo is reported with a rough 95% interval. The build plan's promotion gate is
//! a score rate above 55% against the previous generation.

use bschess::eval::Evaluator;
use bschess::gauntlet::{play_match, round_robin, Participant};
use bschess::nnue::Network;
use bschess::search::SearchLimits;
use std::sync::Arc;

fn load(path: &str) -> Evaluator {
    match Network::load(path) {
        Ok(n) => Evaluator::Nnue(Arc::new(n)),
        Err(e) => {
            eprintln!("could not load network {path}: {e}");
            std::process::exit(2);
        }
    }
}

fn name_of(path: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

fn main() {
    bschess::init();
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut pairs = 100usize;
    let mut depth = 6i32;
    let mut movetime: Option<u64> = None;
    let mut a: Option<String> = None;
    let mut b: Option<String> = None;
    let mut pool: Vec<String> = Vec::new();
    let mut gate = 0.55f64;
    let mut seed = 0xC0FFEEu64;

    let mut i = 0;
    while i < args.len() {
        let flag = args[i].clone();
        let value = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i).cloned().unwrap_or_default()
        };
        match flag.as_str() {
            "--pairs" => pairs = value(&mut i).parse().unwrap_or(pairs),
            "--depth" => depth = value(&mut i).parse().unwrap_or(depth),
            "--movetime" => movetime = value(&mut i).parse().ok(),
            "--a" => a = Some(value(&mut i)),
            "--b" => b = Some(value(&mut i)),
            "--gate" => gate = value(&mut i).parse().unwrap_or(gate),
            "--seed" => seed = value(&mut i).parse().unwrap_or(seed),
            "--pool" => {
                while i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    i += 1;
                    pool.push(args[i].clone());
                }
            }
            "--help" | "-h" => {
                println!(
                    "gauntlet --pairs N [--depth D | --movetime MS] \
                     (--a NET --b NET | --pool NET...) [--gate 0.55] [--seed S]"
                );
                return;
            }
            other => eprintln!("ignoring unknown argument '{other}'"),
        }
        i += 1;
    }

    let limits = match movetime {
        Some(ms) => SearchLimits::movetime_ms(ms),
        None => SearchLimits::depth(depth),
    };

    if !pool.is_empty() {
        // Round robin. The hand evaluation is always in the pool as the fixed
        // Phase 2 yardstick.
        let mut participants = vec![Participant::new(
            format!("hand-d{depth}"),
            Evaluator::Hand,
            limits.clone(),
        )];
        for p in &pool {
            participants.push(Participant::new(name_of(p), load(p), limits.clone()));
        }
        println!(
            "round robin: {} participants, {} games each pairing\n",
            participants.len(),
            pairs * 2
        );
        let table = round_robin(&participants, pairs, seed);
        println!("\n{:<20} {:>6} {:>6} {:>6} {:>8} {:>8}", "engine", "+", "-", "=", "score", "elo");
        let mut rows = table;
        rows.sort_by(|x, y| y.1.rate().partial_cmp(&x.1.rate()).unwrap());
        for (name, s) in rows {
            println!(
                "{:<20} {:>6} {:>6} {:>6} {:>7.1}% {:>+8.0}",
                name,
                s.wins,
                s.losses,
                s.draws,
                100.0 * s.rate(),
                s.elo()
            );
        }
        return;
    }

    let pa = match &a {
        Some(p) => Participant::new(name_of(p), load(p), limits.clone()),
        None => Participant::new(format!("hand-d{depth}"), Evaluator::Hand, limits.clone()),
    };
    let pb = match &b {
        Some(p) => Participant::new(name_of(p), load(p), limits.clone()),
        None => Participant::new(
            format!("hand-d{depth}-baseline"),
            Evaluator::Hand,
            limits.clone(),
        ),
    };

    println!(
        "{} vs {} — {} games ({} openings, both colours), {} threads\n",
        pa.name,
        pb.name,
        pairs * 2,
        pairs,
        rayon::current_num_threads()
    );

    let score = play_match(&pa, &pb, pairs, seed);
    println!("{}", score.summary(&pa.name, &pb.name));

    if score.passes_gate(gate) {
        println!("\nPROMOTE — {:.1}% clears the {:.0}% gate", 100.0 * score.rate(), 100.0 * gate);
    } else {
        println!("\nHOLD — {:.1}% does not clear the {:.0}% gate", 100.0 * score.rate(), 100.0 * gate);
        std::process::exit(1);
    }
}
