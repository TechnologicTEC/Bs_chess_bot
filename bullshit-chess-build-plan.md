# Bullshit Chess — Build Plan v2.0

Companion to `bullshit-chess-spec.md`. The spec defines *what the game is*; this defines *how to build something that plays it well*.

**Decisions locked:** Rust. NNUE evaluation driving an alpha-beta search. Laptop-class hardware, CPU-centric.

---

## 0. Why this shape

The hardware makes the choice. An AlphaZero-style run needs a ~5M parameter ResNet evaluated hundreds of times per move, which on a laptop GPU works out to roughly 20–25k self-play games per day. Reference runs for chess-like games consumed millions to tens of millions of games. That is a months-to-years timeline.

NNUE inverts the ratio. The network is ~200k parameters, trains in minutes, and runs fast enough on CPU that your alpha-beta search stays the bottleneck rather than the GPU. Data generation parallelises trivially across cores. The catch is that you supply the search yourself instead of learning it — but you were building that search anyway as a baseline, and this variant is decided by tactical capture sequences, which is exactly what alpha-beta plus quiescence is good at.

The trade you are making: no "learned from zero knowledge" property, and a somewhat lower theoretical ceiling. In exchange, a working strong engine in weeks.

---

## Phase 1 — Engine core (~4 days)

**Representation.** Bitboards: 12 `u64` piece boards plus occupancy. No board-flipping needed anywhere (see §Phase 3).

**Precompute at startup:**
| Table | Size | Purpose |
|---|---|---|
| 3×3 explosion masks | 64 × `u64` | Rook/bishop detonation regions |
| Knight path masks | 64 × 8 × `u64` | Both intermediates + destination |
| Rook/bishop magics | standard | Ray attacks with blockers |
| Zobrist keys | 12 × 64 | Transposition table |

**Move representation:** pack `(from, to)` into a `u16`. Complete — no promotion or castling flags exist.

Move generation is the easy half of a chess engine here: no check legality, no pin detection, no checkmate search, and provably no stalemate (spec §1.4). Everything is pseudo-legal. The bugs will be in queen sweep enumeration and explosion clipping at board edges.

**Gate: perft.** Write a deliberately slow, obviously-correct Python reference implementation — a day's work — and match node counts at depths 1–4. Do not proceed until they agree. A movegen bug found after two weeks of training is two weeks lost, and you cannot detect one by watching games.

Use `rayon` for parallelism throughout. Your core count is the resource you actually have.

### The branching factor problem

Pawn teleports number `pawns × empty_squares` — around 250 in the opening against maybe 60–100 piece moves. Roughly 300 total, which breaks naive alpha-beta immediately (300⁶ ≈ 7×10¹⁴).

**Restrict teleport generation.** Generate only teleports to squares that are:
- adjacent to either king (creating or denying a detonator),
- on a ray between an enemy slider and your king (blocking),
- adjacent to a friendly rook or bishop that could then be usefully detonated.

That cuts ~250 down to ~15–25 and discards almost nothing of value. Keep full generation available behind a flag for perft correctness.

---

## Phase 2 — Search and hand evaluation (~3 days)

**Components:** iterative deepening, transposition table, killer moves, history heuristic, late move reductions, aspiration windows, and **quiescence search over all captures**.

Quiescence is not optional. One move can swing four or more pieces via a blast or knight sweep, so any fixed-depth search without it will hallucinate constantly.

**Move ordering uses exact deltas.** There is no recapture sequence to model, so standard SEE is meaningless. Instead compute each capture's true material swing directly by AND-ing its affected mask against occupancy. This is strictly better information than a real chess engine has at the ordering stage — use it.

**Be cautious with null-move pruning.** Position volatility is extreme; a quiet-looking position can be lost outright in one move. Verify empirically that it does not cost tactics before keeping it.

Hand evaluation: spec §6. Expect depth 5–7 with restricted teleports.

---

## Phase 3 — Symmetry and the match driver (~2 days)

### Symmetry

No castling and no directional pawn movement means the rules are fully invariant under the dihedral group of the square. Two consequences worth exploiting:

1. **Colour canonicalisation is free.** Swap piece colours and flip side-to-move — no vertical board flip needed, because no piece has a forward direction.
2. **8× data augmentation** on every training position, via 4 rotations × 2 mirrors.

Write and unit-test these transforms in isolation now, before they are load-bearing. A bug here is nearly invisible downstream; it just makes training mysteriously ineffective.

### Two operating modes

The engine needs two distinct front ends, sharing one core.

**Mode A — operator mode (human relay).** You are the interface between your engine and your friend's. You type his move, the engine replies with its own. This is the mode used for real games.

Required commands:

| Command | Effect |
|---|---|
| `new [w\|b]` | Start a game, choosing which colour the engine plays |
| `<move>` | e.g. `e2e4` — apply the opponent's move, then search and reply |
| `go` | Force the engine to move now (for when it has the first move) |
| `board` | Reprint the board |
| `fen` / `setfen <fen>` | Export / import position, for resyncing with your friend |
| `undo` | Take back the last ply — you *will* mistype a move |
| `moves` | List legal moves from the current position |
| `time <ms>` | Set thinking time per move |

After every move — yours or the engine's — print:
- the resulting board,
- **the list of squares emptied by the move**,
- the engine's evaluation, search depth, and principal variation,
- the current ply number.

The emptied-squares line is not a nicety. It is the single most useful diagnostic you will have, for the reason set out in spec §8: a blast clears up to nine squares, and if the two engines disagree about any rule the boards diverge silently and neither player notices for many moves. Printing what was destroyed lets your friend compare against his own output immediately.

`undo` matters more than it looks. Transcription errors are near-certain over a long game, and without takeback a single typo forces you to rebuild the position from FEN by hand.

**Mode B — self-play mode.** Fully automatic, no I/O per move, parallel across cores, writes positions and results to disk for training. This is the mode that consumes 99% of your compute. Keep it entirely separate from operator mode — no shared printing, no shared state, no per-move allocation if you can avoid it.

Build Mode A first, in week 2, even though Mode B is where the value is. It is how you find out whether Phase 2 already beats your friend's bot.

**This is the highest-value checkpoint in the project.** If hand-written alpha-beta already wins, the ML work is about raising a ceiling rather than reaching a baseline, and you can proceed with confidence rather than hope.

---

## Phase 4 — NNUE architecture (~3 days)


**Features.** 768 inputs: 12 piece types × 64 squares. Two accumulators, one per perspective.

**Topology.** `768 → 256` (per perspective, weights shared) → concatenate to 512 → `32` → `32` → `1`. Roughly 200k parameters. ClippedReLU activations.

Optionally add king-bucketing later (indexing features by own-king square, HalfKP-style). Skip it for v1 — it multiplies your feature count and the payoff is uncertain in a variant where king safety works differently.

### Two variant-specific notes that matter

**Skip incremental updates initially.** NNUE's usual efficiency argument is that a chess move changes 1–2 features, so you patch the accumulator instead of recomputing. Here, a single blast can change up to 10 features at once, and captures are exactly the nodes quiescence search spends most of its time in. Fortunately the input is sparse — a full refresh is just summing ~32 columns of 256 values, around 8k additions. That is cheap enough to do at every node. Measure before optimising; you may never need incrementality at all.

**No ply-count input.** The ply cap is handled by the search returning a draw score at the horizon, so the evaluation network never needs to know it. (This requirement applied only to an AlphaZero value head, which had to be Markovian on its own.)

**Precision.** Start in `f32`. Quantise to int16/int8 only if profiling says the evaluation is your bottleneck rather than movegen.

---

## Phase 5 — Training loop

Generational, not continuous. Each generation is a day or less.

```
gen 0: engine uses hand-written eval (Phase 2)

for gen in 1..N:
    self-play M games at fixed depth (6-8) or fixed node count,
        parallel across all cores, randomised openings
    → record every position, its search score, and the game result
    → FILTER: keep only positions whose best move is quiet
    → label = 0.7 * search_score + 0.3 * game_result   (tune lambda)
    → augment 8x via dihedral symmetry
    → train the net in PyTorch (minutes)
    → export weights, plug into the engine
    → gate: new engine vs previous, 200 games, promote at >55%
```

**The quiet-position filter is important here.** Standard NNUE practice is to train evaluation only on positions where nothing is hanging, because quiescence handles tactics. In a variant this volatile, training on tactical positions will teach the net noise. Filter aggressively.

**Volume.** Aim for 5–20M filtered positions per generation. At depth 6 across, say, 8 cores you should manage low-thousands of positions per second, so this is hours, not days. Most of the strength gain arrives in generations 1–4; returns flatten after that.

**Randomised openings** are essential or every game will be identical. Play 4–8 random plies before handing over to the engine, and reject openings where one side is already winning.

---

## Phase 6 — Measuring progress

Fixed gauntlet, ~200 randomised-opening games per promotion:

1. Your existing bot — the actual target
2. The Phase 2 hand-eval engine at fixed depth
3. Three or four frozen previous generations

Track Elo across the pool. **Self-play win rate will lie to you** — a net can improve against itself while getting worse against a different playing style, and you will not notice until you test externally.

---

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Rule mismatch with the opponent bot | Trains a strong player of the wrong game | Perft against independent implementation; share the spec with both engines |
| Board divergence mid-game vs your friend | Game becomes unrecoverable, blame undiagnosable | Print emptied squares after every move; `fen` command for resync (spec §8) |
| Explosion / queen sweep edge cases | Silent, corrupts everything downstream | Spec §7 test suite, before any training code |
| Branching factor kills alpha-beta | No usable baseline at all | Restricted teleport generation, Phase 1 |
| Training on tactical positions | Net learns noise, no strength gain | Quiet-position filter, Phase 5 |
| Policy/board transform bug | Mysteriously ineffective training | Unit-test transforms in isolation, Phase 3 |
| Ply-cap shuffle degeneracy | Engine learns to stall when losing | Material adjudication at ply 200 (spec §4) |

---

## Timeline

| Week | Deliverable |
|---|---|
| 1 | Engine core, perft passing against Python reference |
| 2 | Alpha-beta + hand eval, match driver, **first contest against your existing bot** |
| 3 | NNUE architecture, data generation pipeline, generation 1 trained end to end |
| 4 | Generations 2–4, gating, gauntlet |
| 5+ | Tune search parameters, consider king-bucketing and quantisation |

If you later get access to a real GPU and want the AlphaZero property, everything through Phase 3 is reusable unchanged — the movegen, symmetry transforms and match driver are route-agnostic.
