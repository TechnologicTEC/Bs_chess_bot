//! Mode B — self-play data generation (build plan, Phase 3 / Phase 5).
//!
//!   selfplay --games 20000 --depth 6 --out data/gen1.bin [--net nets/gen0.bin]
//!
//! Runs across every core. Writes fixed-width 32-byte records that
//! `tools/train.py` reads with a single `np.fromfile`.

use bschess::eval::Evaluator;
use bschess::nnue::Network;
use bschess::search::SearchLimits;
use bschess::selfplay::{run, SelfPlayConfig};
use std::path::PathBuf;
use std::sync::Arc;

fn main() {
    bschess::init();
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut cfg = SelfPlayConfig::default();
    let mut out: Option<PathBuf> = None;
    let mut net_path: Option<String> = None;
    let mut threads: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let value = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i).cloned().unwrap_or_default()
        };
        match flag {
            "--games" => cfg.games = value(&mut i).parse().unwrap_or(cfg.games),
            "--depth" => {
                cfg.limits = SearchLimits::depth(value(&mut i).parse().unwrap_or(6));
            }
            "--nodes" => {
                cfg.limits = SearchLimits::nodes(value(&mut i).parse().unwrap_or(20_000));
            }
            "--seed" => cfg.seed = value(&mut i).parse().unwrap_or(cfg.seed),
            "--out" => out = Some(PathBuf::from(value(&mut i))),
            "--net" => net_path = Some(value(&mut i)),
            "--threads" => threads = value(&mut i).parse().ok(),
            "--all-positions" => cfg.quiet_only = false,
            "--help" | "-h" => {
                println!(
                    "selfplay --games N --depth D [--nodes N] [--out FILE] [--net FILE] \
                     [--threads N] [--all-positions] [--seed S]"
                );
                return;
            }
            other => eprintln!("ignoring unknown argument '{other}'"),
        }
        i += 1;
    }

    if let Some(n) = threads {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
    }

    if let Some(p) = &net_path {
        match Network::load(p) {
            Ok(n) => {
                eprintln!("evaluator: nnue ({} parameters) from {p}", n.parameter_count());
                cfg.evaluator = Evaluator::Nnue(Arc::new(n));
            }
            Err(e) => {
                eprintln!("could not load network {p}: {e}");
                std::process::exit(2);
            }
        }
    } else {
        eprintln!("evaluator: hand (generation 0)");
    }

    if let Some(p) = &out {
        if let Some(dir) = p.parent() {
            if !dir.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(dir);
            }
        }
    }

    eprintln!(
        "playing {} games on {} threads, quiet filter {}",
        cfg.games,
        rayon::current_num_threads(),
        if cfg.quiet_only { "on" } else { "off" }
    );

    let start = std::time::Instant::now();
    let stats = match run(&cfg, out.as_deref(), true) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("self-play failed: {e}");
            std::process::exit(1);
        }
    };
    let secs = start.elapsed().as_secs_f64().max(1e-9);

    println!("games         {}", stats.games);
    println!(
        "results       white {} / black {} / draw {}",
        stats.white_wins, stats.black_wins, stats.draws
    );
    println!(
        "avg length    {:.1} plies",
        stats.plies as f64 / stats.games.max(1) as f64
    );
    println!(
        "positions     {} kept of {} seen ({:.1}% passed the quiet filter)",
        stats.positions_kept,
        stats.positions_seen,
        100.0 * stats.positions_kept as f64 / stats.positions_seen.max(1) as f64
    );
    println!(
        "throughput    {:.0} positions/s, {:.2}M nodes/s, {:.1} s total",
        stats.positions_kept as f64 / secs,
        stats.nodes as f64 / secs / 1e6,
        secs
    );
    if let Some(p) = &out {
        println!("wrote         {}", p.display());
    }
}
