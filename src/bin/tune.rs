//! Texel tuning of the hand evaluation weights.
//!
//!   tune --data data/gen1.bin data/gen1_large.bin [--iters 4000] [--sample 0]
//!
//! Spec Â§6 says of its weights: *"Guesses for a first alpha-beta pass. Tune by
//! self-play afterwards."* This is that. Nobody's intuition is calibrated for this
//! variant, and the self-play shards are already a labelled dataset.
//!
//! The evaluation is linear in its ten weights, so fitting them is a small
//! logistic regression: minimise
//!
//!     mean over positions of ( sigmoid(w . f / K) - target )^2
//!
//! where the target blends the search score with the eventual game result, the
//! same label the NNUE trains on.
//!
//! Unlike the network, this costs nothing at run time â€” the tuned weights are the
//! same arithmetic with different constants, so every centipawn of accuracy is
//! kept rather than paid back in search depth.

use bschess::data::Sample;
use bschess::eval::{
    eval_feature_name, eval_features, eval_weights, NUM_EVAL_FEATURES,
};
use bschess::tables::SplitMix64;
use bschess::types::Color;

/// Centipawns per logit when mapping evaluation to win probability. Matches the
/// value `tools/train.py` uses, so the two are fitting the same target.
const K: f32 = 400.0;
/// Weight on the game result versus the search score in the label.
const LAMBDA_RESULT: f32 = 0.3;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

struct Dataset {
    features: Vec<[f32; NUM_EVAL_FEATURES]>,
    targets: Vec<f32>,
}

fn load(paths: &[String], sample: usize, seed: u64) -> Dataset {
    let mut rng = SplitMix64::new(seed);
    let mut features = Vec::new();
    let mut targets = Vec::new();

    for path in paths {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("cannot read {path}: {e}");
                std::process::exit(2);
            }
        };
        let whole = bytes.len() / 32 * 32;
        if whole != bytes.len() {
            eprintln!("  {path}: dropping {} trailing bytes", bytes.len() - whole);
        }
        let n = whole / 32;
        let mut kept = 0;
        for i in 0..n {
            // Subsample deterministically when asked; the fit converges long
            // before it needs three million positions.
            if sample > 0 && rng.below(n) >= sample {
                continue;
            }
            let Some(s) = Sample::decode(&bytes[i * 32..(i + 1) * 32]) else {
                continue;
            };
            let pos = s.to_position();
            if pos.result().is_some() {
                continue;
            }

            // Everything is put in White's frame: the stored score is side-to-move
            // relative, the stored result is already from White's point of view.
            let score_white = if pos.side == Color::White {
                s.score as f32
            } else {
                -(s.score as f32)
            };
            let result_white = s.result as f32;
            let target = (1.0 - LAMBDA_RESULT) * sigmoid(score_white / K)
                + LAMBDA_RESULT * (result_white + 1.0) / 2.0;

            features.push(eval_features(&pos));
            targets.push(target);
            kept += 1;
        }
        println!("  {:<28} {:>9} positions", path, kept);
    }
    Dataset { features, targets }
}

fn loss(d: &Dataset, w: &[f32; NUM_EVAL_FEATURES], range: std::ops::Range<usize>) -> f32 {
    let mut total = 0.0;
    for i in range.clone() {
        let mut e = 0.0;
        for (f, wj) in d.features[i].iter().zip(w.iter()) {
            e += f * wj;
        }
        let diff = sigmoid(e / K) - d.targets[i];
        total += diff * diff;
    }
    total / range.len().max(1) as f32
}

fn main() {
    bschess::init();
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut paths: Vec<String> = Vec::new();
    let mut iters = 4000usize;
    let mut sample = 0usize;
    let mut lr = 4.0f32;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                while i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    i += 1;
                    paths.push(args[i].clone());
                }
            }
            "--iters" => {
                i += 1;
                iters = args[i].parse().unwrap_or(iters);
            }
            "--sample" => {
                i += 1;
                sample = args[i].parse().unwrap_or(0);
            }
            "--lr" => {
                i += 1;
                lr = args[i].parse().unwrap_or(lr);
            }
            other => eprintln!("ignoring unknown argument '{other}'"),
        }
        i += 1;
    }
    if paths.is_empty() {
        eprintln!("usage: tune --data FILE... [--iters N] [--sample N] [--lr X]");
        std::process::exit(2);
    }

    println!("loading:");
    let t0 = std::time::Instant::now();
    let d = load(&paths, sample, 0x7005EED);
    let n = d.features.len();
    if n < 1000 {
        eprintln!("only {n} usable positions â€” need more data");
        std::process::exit(2);
    }
    // Hold out a slice for an honest read. Ten parameters cannot overfit three
    // million positions, but a validation number costs nothing and catches a
    // broken objective.
    let n_val = (n / 20).clamp(1, 200_000);
    let train = n_val..n;
    let val = 0..n_val;
    println!(
        "  {n} positions in {:.1}s  ({} train / {n_val} validation)",
        t0.elapsed().as_secs_f32(),
        n - n_val
    );

    let start = eval_weights();
    println!(
        "\nstarting loss   train {:.6}  validation {:.6}",
        loss(&d, &start, train.clone()),
        loss(&d, &start, val.clone())
    );

    // Adam. The weights span 25 to 1000 centipawns, so a single global step size
    // would either crawl on the queen or diverge on the pawn; per-parameter
    // adaptive rates handle the spread without hand-scaled features.
    let mut w = start;
    let mut m = [0f32; NUM_EVAL_FEATURES];
    let mut v = [0f32; NUM_EVAL_FEATURES];
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);

    for step in 1..=iters {
        let mut grad = [0f32; NUM_EVAL_FEATURES];
        for i in train.clone() {
            let f = &d.features[i];
            let mut e = 0.0;
            for j in 0..NUM_EVAL_FEATURES {
                e += f[j] * w[j];
            }
            let s = sigmoid(e / K);
            // d/dw of (s - t)^2 with s = sigmoid(w.f/K)
            let g = 2.0 * (s - d.targets[i]) * s * (1.0 - s) / K;
            for j in 0..NUM_EVAL_FEATURES {
                grad[j] += g * f[j];
            }
        }
        let inv = 1.0 / train.len() as f32;
        let (c1, c2) = (1.0 - b1.powi(step as i32), 1.0 - b2.powi(step as i32));
        for (((wj, mj), vj), gj) in w
            .iter_mut()
            .zip(m.iter_mut())
            .zip(v.iter_mut())
            .zip(grad.iter())
        {
            let g = gj * inv;
            *mj = b1 * *mj + (1.0 - b1) * g;
            *vj = b2 * *vj + (1.0 - b2) * g * g;
            *wj -= lr * (*mj / c1) / ((*vj / c2).sqrt() + eps);
        }

        if step.is_multiple_of((iters / 10).max(1)) {
            println!(
                "  step {step:>5}   train {:.6}  validation {:.6}",
                loss(&d, &w, train.clone()),
                loss(&d, &w, val.clone())
            );
        }
    }

    println!(
        "\nfinal loss      train {:.6}  validation {:.6}",
        loss(&d, &w, train.clone()),
        loss(&d, &w, val.clone())
    );

    println!("\n{:<20} {:>10} {:>10} {:>10}", "term", "before", "after", "change");
    for j in 0..NUM_EVAL_FEATURES {
        println!(
            "{:<20} {:>10.0} {:>10.0} {:>+10.0}",
            eval_feature_name(j),
            start[j],
            w[j].round(),
            (w[j] - start[j]).round()
        );
    }

    println!("\n--- paste into src/eval.rs ---");
    println!("pub const TUNED_V2_WEIGHTS: HandWeights = [");
    for chunk in w.chunks(10) {
        let row: Vec<String> = chunk.iter().map(|x| format!("{:.0}", x.round())).collect();
        println!("    {},", row.join(", "));
    }
    println!("];");
}
