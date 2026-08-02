//! Measuring progress (build plan, Phase 6).
//!
//! Self-play win rate will lie to you â€” a net can improve against itself while
//! getting worse against a different playing style. Everything here plays engines
//! against *other* engines over a fixed, randomised-opening book, with each
//! opening played twice so both sides get it.

use crate::eval::Evaluator;
use crate::position::{GameResult, Position};
use crate::search::{SearchLimits, SearchOptions, Searcher};
use crate::selfplay::{random_opening, SelfPlayConfig};
use crate::tables::SplitMix64;
use crate::types::Color;

#[derive(Clone)]
pub struct Participant {
    pub name: String,
    pub evaluator: Evaluator,
    pub limits: SearchLimits,
    pub options: SearchOptions,
}

impl Participant {
    pub fn new(name: impl Into<String>, evaluator: Evaluator, limits: SearchLimits) -> Participant {
        let options = SearchOptions {
            evaluator: evaluator.clone(),
            tt_megabytes: 16,
            verbose: false,
            ..Default::default()
        };
        Participant {
            name: name.into(),
            evaluator,
            limits,
            options,
        }
    }
    fn searcher(&self) -> Searcher {
        Searcher::new(self.options.clone())
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct MatchScore {
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

impl MatchScore {
    pub fn games(&self) -> u32 {
        self.wins + self.losses + self.draws
    }
    /// Score rate in [0, 1] â€” a draw counts a half.
    pub fn rate(&self) -> f64 {
        let g = self.games();
        if g == 0 {
            return 0.5;
        }
        (self.wins as f64 + 0.5 * self.draws as f64) / g as f64
    }
    /// Elo difference implied by the score rate. Saturates rather than diverging
    /// on a clean sweep.
    pub fn elo(&self) -> f64 {
        let g = self.games().max(1) as f64;
        let r = self.rate().clamp(0.5 / g, 1.0 - 0.5 / g);
        -400.0 * ((1.0 / r) - 1.0).log10()
    }
    /// Rough 95% confidence half-width on the Elo estimate.
    pub fn elo_margin(&self) -> f64 {
        let g = self.games().max(1) as f64;
        let r = self.rate();
        let se = ((r * (1.0 - r)) / g).sqrt().max(1e-6);
        // d(Elo)/dr at r, via the logistic inverse.
        let d = 400.0 / (r.clamp(0.01, 0.99) * (1.0 - r.clamp(0.01, 0.99)) * 10f64.ln());
        1.96 * se * d
    }
    pub fn merge(&mut self, o: &MatchScore) {
        self.wins += o.wins;
        self.losses += o.losses;
        self.draws += o.draws;
    }
    pub fn summary(&self, a: &str, b: &str) -> String {
        format!(
            "{a} vs {b}: +{} -{} ={}  ({:.1}%)  Elo {:+.0} Â± {:.0}",
            self.wins,
            self.losses,
            self.draws,
            100.0 * self.rate(),
            self.elo(),
            self.elo_margin()
        )
    }
    /// Promotion gate from the build plan: a new generation must beat the previous
    /// one at better than 55%.
    pub fn passes_gate(&self, threshold: f64) -> bool {
        self.rate() > threshold
    }
}

/// Play one game between two participants from a given opening.
pub fn play_game(
    white: &Participant,
    black: &Participant,
    opening: &Position,
) -> GameResult {
    let mut pos = *opening;
    let mut sw = white.searcher();
    let mut sb = black.searcher();
    sw.reset();
    sb.reset();
    loop {
        if let Some(r) = pos.result() {
            return r;
        }
        let (searcher, limits) = if pos.side == Color::White {
            (&mut sw, &white.limits)
        } else {
            (&mut sb, &black.limits)
        };
        let r = searcher.search(&pos, limits);
        if r.best.is_none() {
            return pos.adjudicate();
        }
        pos.make_move(r.best);
    }
}

/// Play `pairs * 2` games â€” every opening once from each side â€” in parallel.
/// Returns the score from `a`'s point of view.
///
/// Progress goes to stderr as pairs complete. A match can run for hours when one
/// side is an NNUE, and silence for that long is indistinguishable from a hang.
pub fn play_match(a: &Participant, b: &Participant, pairs: usize, seed: u64) -> MatchScore {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    let opening_cfg = SelfPlayConfig {
        evaluator: a.evaluator.clone(),
        ..Default::default()
    };
    let openings: Vec<Position> = (0..pairs)
        .map(|i| {
            let mut rng = SplitMix64::new(seed ^ (i as u64).wrapping_mul(0xA24BAED4963EE407));
            random_opening(&opening_cfg, &mut rng)
        })
        .collect();

    let done = AtomicU64::new(0);
    let running = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    let start = std::time::Instant::now();

    let scores: Vec<MatchScore> = openings
        .par_iter()
        .with_max_len(1)
        .map(|opening| {
            let mut s = MatchScore::default();
            for a_is_white in [true, false] {
                let (w, bl) = if a_is_white { (a, b) } else { (b, a) };
                let result = play_game(w, bl, opening);
                let a_color = if a_is_white { Color::White } else { Color::Black };
                match result {
                    GameResult::Draw => s.draws += 1,
                    GameResult::Win(c) if c == a_color => s.wins += 1,
                    GameResult::Win(_) => s.losses += 1,
                }
            }
            running[0].fetch_add(s.wins as u64, Ordering::Relaxed);
            running[1].fetch_add(s.losses as u64, Ordering::Relaxed);
            running[2].fetch_add(s.draws as u64, Ordering::Relaxed);

            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            let step = (pairs / 20).max(1);
            if n.is_multiple_of(step as u64) || n == pairs as u64 {
                let elapsed = start.elapsed().as_secs_f64();
                let eta = elapsed / n as f64 * (pairs as f64 - n as f64);
                eprintln!(
                    "  {:>4}/{} pairs  +{} -{} ={}  [{:.0}s elapsed, ~{:.0}s left]",
                    n,
                    pairs,
                    running[0].load(Ordering::Relaxed),
                    running[1].load(Ordering::Relaxed),
                    running[2].load(Ordering::Relaxed),
                    elapsed,
                    eta
                );
            }
            s
        })
        .collect();

    let mut total = MatchScore::default();
    for s in &scores {
        total.merge(s);
    }
    total
}

/// Round-robin over a pool, printing a table. The pool should hold the actual
/// target opponent, the Phase 2 hand-eval engine, and three or four frozen
/// previous generations.
pub fn round_robin(pool: &[Participant], pairs: usize, seed: u64) -> Vec<(String, MatchScore)> {
    let mut totals: Vec<MatchScore> = vec![MatchScore::default(); pool.len()];
    for i in 0..pool.len() {
        for j in i + 1..pool.len() {
            let s = play_match(&pool[i], &pool[j], pairs, seed ^ ((i * 31 + j) as u64));
            println!("{}", s.summary(&pool[i].name, &pool[j].name));
            totals[i].merge(&s);
            totals[j].merge(&MatchScore {
                wins: s.losses,
                losses: s.wins,
                draws: s.draws,
            });
        }
    }
    pool.iter()
        .map(|p| p.name.clone())
        .zip(totals)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elo_is_zero_at_even_score() {
        let s = MatchScore {
            wins: 10,
            losses: 10,
            draws: 4,
        };
        assert!(s.elo().abs() < 1e-9);
        assert!((s.rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn elo_saturates_rather_than_diverging() {
        let s = MatchScore {
            wins: 20,
            losses: 0,
            draws: 0,
        };
        assert!(s.elo().is_finite() && s.elo() > 300.0);
    }

    #[test]
    fn a_deeper_engine_does_not_lose_to_a_shallower_one() {
        let deep = Participant::new("d4", Evaluator::Hand, SearchLimits::depth(4));
        let shallow = Participant::new("d1", Evaluator::Hand, SearchLimits::depth(1));
        let s = play_match(&deep, &shallow, 3, 11);
        assert_eq!(s.games(), 6);
        assert!(
            s.rate() >= 0.5,
            "depth 4 scored {:.2} against depth 1",
            s.rate()
        );
    }
}
