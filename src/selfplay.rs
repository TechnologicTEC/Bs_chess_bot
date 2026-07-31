//! Mode B â€” self-play data generation (build plan, Phase 3 / Phase 5).
//!
//! Fully automatic, no I/O per move, parallel across cores. Kept deliberately
//! separate from operator mode: no shared printing, no shared state, and the
//! per-game allocations are bounded by the ply cap.

use crate::data::Sample;
use crate::eval::{Evaluator, MATE_THRESHOLD};
use crate::movegen::{generate, is_capture, GenMode, MoveList};
use crate::position::{GameResult, Position};
use crate::search::{SearchLimits, SearchOptions, Searcher};
use crate::tables::SplitMix64;
use crate::types::*;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone)]
pub struct SelfPlayConfig {
    pub games: usize,
    pub limits: SearchLimits,
    pub evaluator: Evaluator,
    /// Random plies played before the engines take over. Essential â€” without it
    /// every game is identical.
    pub opening_plies_min: usize,
    pub opening_plies_max: usize,
    /// Reject an opening if a quick search already puts one side this far ahead.
    pub opening_reject_cp: i32,
    /// Keep only positions whose best move is quiet. In a variant this volatile,
    /// training on tactical positions teaches the net noise.
    pub quiet_only: bool,
    pub seed: u64,
    pub tt_megabytes: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            games: 1000,
            limits: SearchLimits::depth(6),
            evaluator: Evaluator::Hand,
            opening_plies_min: 4,
            opening_plies_max: 8,
            opening_reject_cp: 400,
            quiet_only: true,
            seed: 0x5eed,
            // Measured at depth 6 on 22 threads: 8 MB gives 53 positions/s,
            // 16 MB 66, 32 MB 78, 64 MB 76. Starving the table costs far more in
            // re-searched nodes than it saves in cache pressure — 1 MB is four
            // times slower than 32 MB. Costs threads x 32 MB of RAM, so ~700 MB
            // on a 22-core box; drop it with --tt if that is too much.
            tt_megabytes: 32,
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct SelfPlayStats {
    pub games: u64,
    pub white_wins: u64,
    pub black_wins: u64,
    pub draws: u64,
    pub positions_seen: u64,
    pub positions_kept: u64,
    pub nodes: u64,
    pub plies: u64,
}

impl SelfPlayStats {
    pub fn merge(&mut self, o: &SelfPlayStats) {
        self.games += o.games;
        self.white_wins += o.white_wins;
        self.black_wins += o.black_wins;
        self.draws += o.draws;
        self.positions_seen += o.positions_seen;
        self.positions_kept += o.positions_kept;
        self.nodes += o.nodes;
        self.plies += o.plies;
    }
}

/// Play random plies from the start position, rejecting lopsided openings.
pub fn random_opening(cfg: &SelfPlayConfig, rng: &mut SplitMix64) -> Position {
    let span = cfg.opening_plies_max.saturating_sub(cfg.opening_plies_min) + 1;
    for _ in 0..24 {
        let n = cfg.opening_plies_min + rng.below(span);
        let mut pos = Position::startpos();
        let mut ok = true;
        for _ in 0..n {
            if pos.result().is_some() {
                ok = false;
                break;
            }
            let mut list = MoveList::new();
            generate(&pos, GenMode::Search, &mut list);
            if list.is_empty() {
                ok = false;
                break;
            }
            pos.make_move(list.moves[rng.below(list.len())]);
        }
        if !ok || pos.result().is_some() {
            continue;
        }
        let mut probe = Searcher::new(SearchOptions {
            evaluator: cfg.evaluator.clone(),
            tt_megabytes: 1,
            verbose: false,
            ..Default::default()
        });
        let r = probe.search(&pos, &SearchLimits::depth(3));
        if r.score.abs() <= cfg.opening_reject_cp {
            return pos;
        }
    }
    Position::startpos()
}

/// Play one game to completion, returning every position visited after the
/// opening plus the final result.
pub fn play_game(cfg: &SelfPlayConfig, rng: &mut SplitMix64, searcher: &mut Searcher)
    -> (Vec<Sample>, GameResult, SelfPlayStats)
{
    let mut pos = random_opening(cfg, rng);
    let mut samples: Vec<Sample> = Vec::with_capacity(128);
    let mut stats = SelfPlayStats::default();
    searcher.reset();

    let result = loop {
        if let Some(r) = pos.result() {
            break r;
        }
        let r = searcher.search(&pos, &cfg.limits);
        stats.nodes += r.nodes;
        stats.plies += 1;
        if r.best.is_none() {
            break pos.adjudicate();
        }

        stats.positions_seen += 1;
        let quiet = !is_capture(&pos, r.best);
        // Scores near the terminal bounds carry no evaluation signal â€” they are
        // search facts, not position judgements.
        let usable = r.score.abs() < MATE_THRESHOLD;
        if usable && (quiet || !cfg.quiet_only) {
            samples.push(Sample::from_position(&pos, r.score, quiet));
            stats.positions_kept += 1;
        }

        pos.make_move(r.best);
    };

    for s in samples.iter_mut() {
        s.set_result(result);
    }
    stats.games = 1;
    match result {
        GameResult::Draw => stats.draws += 1,
        GameResult::Win(Color::White) => stats.white_wins += 1,
        GameResult::Win(Color::Black) => stats.black_wins += 1,
    }
    (samples, result, stats)
}

/// Run `cfg.games` games across all cores, streaming records to `out`.
pub fn run(cfg: &SelfPlayConfig, out: Option<&Path>, progress: bool) -> std::io::Result<SelfPlayStats> {
    use rayon::prelude::*;

    let writer = out
        .map(|p| std::fs::File::create(p).map(std::io::BufWriter::new))
        .transpose()?;
    let writer = writer.map(std::sync::Mutex::new);
    let done = AtomicU64::new(0);
    let total = cfg.games as u64;

    // `with_max_len(1)` keeps rayon from batching games into contiguous chunks.
    // Game lengths vary by an order of magnitude here — a deep search can run to
    // the 200-ply cap — so a chunk holding a few long games becomes a straggler
    // that the whole run waits on. One game per work item lets stealing even it
    // out. Measured at depth 6: 23 -> ~200 positions/s.
    let chunks: Vec<usize> = (0..cfg.games).collect();
    let stats: Vec<SelfPlayStats> = chunks
        .par_iter()
        .with_max_len(1)
        .map(|&i| {
            let mut rng = SplitMix64::new(cfg.seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let mut searcher = Searcher::new(SearchOptions {
                evaluator: cfg.evaluator.clone(),
                tt_megabytes: cfg.tt_megabytes,
                verbose: false,
                ..Default::default()
            });
            let (samples, _, stats) = play_game(cfg, &mut rng, &mut searcher);

            if let Some(w) = writer.as_ref() {
                let mut buf = Vec::with_capacity(samples.len() * 32);
                for s in &samples {
                    buf.extend_from_slice(&s.encode());
                }
                let mut guard = w.lock().unwrap();
                let _ = guard.write_all(&buf);
            }

            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            if progress && (n.is_multiple_of(50) || n == total) {
                eprintln!("  {n}/{total} games");
            }
            stats
        })
        .collect();

    if let Some(w) = writer {
        w.lock().unwrap().flush()?;
    }

    let mut merged = SelfPlayStats::default();
    for s in &stats {
        merged.merge(s);
    }
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_game_terminates_and_labels_samples() {
        let cfg = SelfPlayConfig {
            games: 2,
            limits: SearchLimits::depth(2),
            ..Default::default()
        };
        let mut rng = SplitMix64::new(7);
        let mut s = Searcher::new(SearchOptions {
            tt_megabytes: 1,
            ..Default::default()
        });
        let (samples, result, stats) = play_game(&cfg, &mut rng, &mut s);
        assert_eq!(stats.games, 1);
        assert!(stats.plies > 0);
        let expect = match result {
            GameResult::Draw => 0,
            GameResult::Win(Color::White) => 1,
            GameResult::Win(Color::Black) => -1,
        };
        for x in &samples {
            assert_eq!(x.result, expect);
            assert!(x.is_quiet());
        }
    }
}
