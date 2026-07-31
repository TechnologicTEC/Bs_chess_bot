# Bullshit Chess engine

A Rust engine for the variant defined in [`bullshit-chess-spec.md`](bullshit-chess-spec.md),
built to [`bullshit-chess-build-plan.md`](bullshit-chess-build-plan.md): bitboard move
generation, alpha-beta search with quiescence, and NNUE evaluation trained by self-play.

The one thing to internalise before reading any code: **every pseudo-legal move is legal**
(spec §1.1) and **a side to move always has one** (spec §1.4). There is no check, no pin
detection, no checkmate search and no stalemate. If the move generator ever returns an
empty list, that is a bug.

---

## Quick start

```sh
cargo build --release

# Operator mode — you relay your opponent's moves, the engine replies.
cargo run --release --bin bschess

# Verify the rules against the independent Python reference.
python tools/perft_gate.py

# Generate data, train a net, gate it, promote it.
python tools/generation.py --gen 1 --games 20000 --depth 6
```

Requires Rust (stable) and, for training only, Python with `torch` and `numpy`.

---

## Playing a game against your opponent's engine

`bschess` is Mode A: you are the interface between this engine and your opponent's.

```
> new b            # engine plays black; you type white's moves
> e2e4             # your opponent's move, applied; engine searches and replies
> undo             # take back a ply — you will mistype a move
> fen              # print FEN, to resync if you suspect divergence
> setfen <fen>     # load a position
> moves            # list legal moves
> time 5000        # thinking time per move, milliseconds
> net nets/current.bin
```

After every move, yours or the engine's, it prints:

```
white plays d1d4   [DETONATION]
  destroyed: Qc4 Rd1 (mover) pd4
  emptied:   c4 d1 d4
  ply 13   black to move
  fen 4k2r/8/8/8/8/8/8/4K3 b - - 13 7
```

**The `destroyed` and `emptied` lines are the point.** A single blast clears up to nine
squares. If the two engines disagree about any rule — blast clipping at an edge, whether a
knight's intermediate square was occupied, where a queen sweep stopped — the boards diverge
silently and neither player notices for many moves (spec §8). Printing what was destroyed
lets your opponent compare against their own output immediately, on the move it happens.

`undo` matters more than it looks. Transcription errors are near-certain over a long game,
and without takeback one typo forces you to rebuild the position from FEN by hand.

---

## The Phase 1 gate

`ref/reference.py` is a deliberately slow, obviously-correct implementation written in a
completely different style — a mailbox board of `(rank, file)` tuples, no bitboards, no
precomputed tables, no cleverness anywhere. It exists for one purpose: to disagree with the
engine if the engine is wrong.

```sh
python tools/perft_gate.py
```

Both sides read the same suite files, so neither can drift onto a different set of
positions. Current status:

| Check | Result |
|---|---|
| Start position, depths 1–3 | 292 / 84,165 / 24,169,988 — exact agreement |
| Start position, depth 4 | 6,972,120,956 — agrees on every one of the 292 root moves |
| Curated suite (`tests/perft_suite.txt`) | 49 / 49 counts agree |
| Random suite (`tests/perft_random_suite.txt`) | 322 / 322 counts agree, 181M nodes |
| Spec §7 checklist (`tests/rules.rs`) | 31 / 31 |

The curated suite covers blast clipping at edges, knight paths crossing kings, queen sweep
stops and every terminal condition. The random suite is 150 positions drawn from real play —
curated positions only prove the rules you thought to test.

Regenerate the random suite with `target/release/perft --random 150 6 <seed>`. Depth is
chosen per position from a movegen-call budget, so sparse boards go deep and crowded ones
stay shallow.

Depth 4 from the start position is 24M subtree expansions, which is hours single-threaded in
Python. `--jobs` fans the 292 root subtrees across processes and brings it down to minutes:

```sh
python ref/reference.py --depth 4 --divide --jobs 0 > ref4.txt   # 0 = one per core
target/release/perft 4 --divide > engine4.txt
diff ref4.txt engine4.txt
```

**Do not proceed past a mismatch.** A movegen bug found after two weeks of training is two
weeks lost, and you cannot detect one by watching games.

---

## Design notes

### The branching factor

Pawn teleports number `pawns × empty_squares` — 256 of the start position's 292 moves. Naive
alpha-beta is hopeless at that width, so `GenMode::Search` restricts teleport destinations to:

1. squares adjacent to either king (creating or denying a detonator),
2. squares between an enemy slider and our king that the slider can currently reach,
3. squares a friendly rook or bishop could reach where the resulting blast catches an enemy
   non-pawn — manufacturing a detonator, spec §5.2,
4. a small spread sample of neutral parking squares, so a pawn standing next to our own king
   (a liability, spec §5.1) can always be evacuated.

That is ~250 destinations down to roughly 15–25. `GenMode::All` keeps the full set, and perft
always uses it — the restriction is a search heuristic, not a rule.

### Copy-make, not undo

One move can destroy nine pieces. A delta-undo record for that is a bug farm; `Position` is
136 bytes and `Copy`, so the search just keeps a snapshot per ply.

### Exact deltas instead of SEE

There is no recapture sequence in this variant, so standard SEE is meaningless (spec §5.6).
`eval::move_delta` computes the true material swing of any move with one mask intersection.
That is strictly better information than a real chess engine has at ordering time, and the
search uses it for both ordering and quiescence pruning.

### Quiescence is not optional

One move can swing four or more pieces via a blast or a knight sweep. Any fixed-depth search
without quiescence hallucinates constantly.

### Null-move pruning is off by default

Position volatility is extreme — a quiet-looking position can be lost outright in one move.
The code is there (`SearchOptions::use_null_move`); measure that it does not cost tactics
before turning it on.

### Symmetry is free

No castling and no directional pawn movement means the rules are fully invariant under the
dihedral group of the square. So colour canonicalisation needs no board flip (just swap piece
colours and the side to move), and every training position augments 8×.

`src/symmetry.rs` is tested against move generation, `make_move` and evaluation rather than
against itself — a bug there is otherwise invisible, showing up only as training that
mysteriously does not work.

### No ply-count network input

The build plan drops the spec §4 requirement that the evaluator see the ply count. That
requirement applied to an AlphaZero value head, which had to be Markovian on its own. Here
`Position::result` adjudicates at the horizon by material, so the search handles the cap and
the network never needs to know the ply.

---

## Performance

On a 22-core laptop, release build:

| Measurement | Value |
|---|---|
| Search from the start position | depth 7 in ~2 s, 3.9M nodes/s (one thread) |
| Self-play, depth 5 | ~260 positions/s, 0.25 s/game across all cores |
| Self-play, depth 6 | ~83 positions/s, 1.3 s/game across all cores |
| Perft, start position depths 1–4 | 6.97e9 nodes in 1.1 s (all cores) |

Self-play throughput is very sensitive to transposition table size, and in the
non-obvious direction. Measured at depth 6 across 22 threads:

| TT per game | Positions/s |
|---|---|
| 1 MB | 16 |
| 8 MB | 53 |
| 32 MB | **78** |
| 64 MB | 76 |

Starving the table costs far more in re-searched nodes than it saves in cache
pressure — 1 MB is nearly five times slower than 32 MB, even though 22 threads at
32 MB is a 700 MB working set. The default is 32 MB; lower it with `--tt` if
memory is tight. This one default was worth a 1.5x speedup at depth 6.

Evaluation is `f32` with a full accumulator refresh at every node. NNUE's usual incremental
update is a bad fit here — one blast changes up to ten features, and captures are exactly
where quiescence spends its time — and a refresh is only ~32 columns of 256 floats. Measure
before optimising; quantisation to int16/int8 is only worth it if profiling says evaluation,
rather than movegen, is the bottleneck.

---

## The training loop

```
gen 0: the hand evaluation (spec §6)

for gen in 1..N:
    self-play M games at fixed depth, all cores, randomised openings
      -> keep only positions whose best move is quiet
      -> label = 0.7 * search_score + 0.3 * game_result, in win-probability space
      -> augment 8x via the dihedral group
      -> train in PyTorch
      -> verify the exported weights evaluate identically in Rust and PyTorch
      -> gate vs the previous generation, promote above 55%
```

`python tools/generation.py --gen N` runs all of it. Individually:

```sh
target/release/selfplay  --games 20000 --depth 6 --out data/gen1.bin [--net nets/gen0.bin]
python tools/train.py    --data data/gen1.bin --out nets/gen1.bin --epochs 20
python tools/verify_net.py --net nets/gen1.bin
target/release/gauntlet  --pairs 100 --depth 6 --a nets/gen1.bin --b hand
```

Aim for 5–20M filtered positions per generation. Most of the strength gain arrives in
generations 1–4; returns flatten after that.

### Two things that will bite you

**The quiet filter matters here more than in normal chess.** Training evaluation on tactical
positions teaches the net noise, because quiescence is already handling the tactics. Self-play
tags each record and keeps only the quiet ones by default.

**Self-play win rate will lie to you.** A net can improve against itself while getting worse
against a different playing style. `gauntlet` always plays engines against *other* engines,
over a fixed randomised opening book, each opening played twice so both sides get it. Keep the
hand evaluation and three or four frozen previous generations in the pool:

```sh
target/release/gauntlet --pairs 50 --depth 6 --pool nets/gen1.bin nets/gen2.bin nets/gen3.bin
```

### Weight serialisation

`tools/verify_net.py` is not optional ceremony. The PyTorch-to-Rust seam is the one place in
this project where a bug is completely silent: transpose a matrix and nothing crashes, nothing
fails a test, the engine just plays badly and you spend a week blaming the training data.

Note that `Nnue.forward` in the trainer returns a **win-probability logit**, not centipawns;
the centipawn scale is folded into the final layer at export. Training the output directly in
centipawns puts every gradient through a 1/400 divisor and the network never moves off zero.

---

## Layout

| Path | Phase | What |
|---|---|---|
| `src/types.rs` | 1 | squares, colours, pieces, the 12-bit move |
| `src/tables.rs` | 1 | blast masks, knight paths, rays, between masks, Zobrist |
| `src/position.rs` | 1 | board, `make_move`, FEN, terminal conditions |
| `src/movegen.rs` | 1 | spec §2 generation, restricted teleports, exact capture effects |
| `src/perft.rs` | 1 | the gate |
| `src/eval.rs` | 2 | spec §6 hand evaluation, exact move deltas |
| `src/search.rs` | 2 | alpha-beta, TT, killers, history, LMR, quiescence |
| `src/symmetry.rs` | 3 | dihedral transforms, colour canonicalisation |
| `src/selfplay.rs` | 3 | Mode B data generation |
| `src/nnue.rs` | 4 | 768-256×2-32-32-1 |
| `src/data.rs` | 5 | the 32-byte training record |
| `src/gauntlet.rs` | 6 | matches, Elo, promotion gate |
| `src/bin/bschess.rs` | 3 | Mode A operator CLI |
| `ref/reference.py` | 1 | the independent implementation |
| `tools/` | 1, 5, 6 | gate, trainer, weight check, generation driver |

---

## Testing

```sh
cargo test                 # 76 tests; debug build keeps the make_move assertions live
cargo test --release
python tools/perft_gate.py
```

Worth knowing about three of them:

- `movegen::capture_effects_agrees_with_make_move` — a property check over random play that
  the ordering-time capture mask always equals what `make_move` actually destroys. If those
  ever diverge, move ordering is lying to the search.
- `symmetry::*_commutes_with_symmetry` — the transforms are checked against movegen and
  `make_move`, not against themselves.
- `rules::perft_matches_the_independent_reference_implementation` — the reference node counts
  frozen in, so the gate is a permanent regression test rather than a one-time check.

---

## Not done

- The Phase 3 checkpoint the plan calls the highest-value one — *does the hand-written
  alpha-beta already beat your opponent's bot?* — needs the opponent. Everything to run it is
  here; nothing substitutes for playing the games.
- A real training run. The pipeline is exercised end to end, but only on 400 self-play games,
  which is a smoke test and not a generation. Gen 1 at that volume scores 51.9% against the
  hand evaluation — parity within the error bar, which is the expected result for a net
  trained to imitate the evaluation it was generated from.
- King-bucketing (HalfKP-style) and int8/int16 quantisation. Both are Phase 5+ in the plan and
  neither is worth doing before a generation actually gates.
