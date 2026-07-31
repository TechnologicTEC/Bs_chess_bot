# Bullshit Chess — Engine Specification v1.0

Authoritative rules reference. Both engines should be validated against this document and its test suite.

Board: standard 8×8, standard starting position. Coordinates `(r, c)` with `r` = rank index 0–7, `c` = file index 0–7.

---

## 1. General principles

1.1. There is **no concept of check, checkmate, or legal-move filtering for king safety**. Every pseudo-legal move is legal. A player may move their king onto a square where it will be captured, may blow up their own king, and may capture their own pieces.

1.2. A move is fully described by the pair `(from, to)`. There are no promotions, no castling, and no en passant, so `(from, to)` is a complete and unambiguous encoding for every move in the game. Total move space: 64 × 64 = 4096.

1.3. The game ends immediately when a king leaves the board, when only kings and pawns remain, or at the ply cap.

---

## 2. Piece movement

### 2.1 Pawn — Teleport

- Moves to **any empty square on the board**, regardless of distance or direction.
- **Cannot capture.** A pawn may never move onto an occupied square.
- Cannot "teleport" to the square it already occupies (no null move).
- Does not promote, ever, on any rank.
- **Blocks sliding pieces** normally (rook, bishop, queen rays).
- Can be captured by any enemy or friendly capture mechanism (explosions, knight sweeps, queen sweeps, king captures).

Move count: `(number of own pawns) × (number of empty squares)`.

### 2.2 Knight — Sweeper

- Moves in the standard 8 L-shapes.
- The path is defined as **two steps along the major axis, then one step perpendicular**. Both intermediate squares plus the destination are part of the path.
- **Captures every piece on all three path squares**, friendly and enemy alike.
- Cannot be blocked. There is no such thing as an illegal knight move due to occupancy.
- The knight **survives** and lands on the destination square.

Path table (offsets from origin):

| Move `(dr, dc)` | Intermediate 1 | Intermediate 2 | Destination |
|---|---|---|---|
| `(+2, +1)` | `(+1, 0)` | `(+2, 0)` | `(+2, +1)` |
| `(+2, -1)` | `(+1, 0)` | `(+2, 0)` | `(+2, -1)` |
| `(-2, +1)` | `(-1, 0)` | `(-2, 0)` | `(-2, +1)` |
| `(-2, -1)` | `(-1, 0)` | `(-2, 0)` | `(-2, -1)` |
| `(+1, +2)` | `(0, +1)` | `(0, +2)` | `(+1, +2)` |
| `(-1, +2)` | `(0, +1)` | `(0, +2)` | `(-1, +2)` |
| `(+1, -2)` | `(0, -1)` | `(0, -2)` | `(+1, -2)` |
| `(-1, -2)` | `(0, -1)` | `(0, -2)` | `(-1, -2)` |

Note that move pairs share intermediates. Precompute 64 × 8 path masks as `u64` bitboards.

A knight can capture a king by **passing over it**, not only by landing on it.

### 2.3 Rook and Bishop — One-shot detonators

- Move along their standard rays. **Blocked normally** — they cannot slide past an occupied square.
- Moving to an **empty** square is an ordinary quiet move with no side effects.
- Moving onto an **occupied** square triggers a detonation:
  - The 3×3 region **centred on the destination square** is cleared. This is the destination plus its up-to-8 neighbours, clipped at board edges.
  - **Every piece in that region is removed**, of either colour, of any type, including kings.
  - **The moving piece is also removed.** It does not survive its own explosion (atomic-chess semantics).
  - Net effect: the mover is spent, and up to 9 squares are emptied.
- A rook or bishop may capture a friendly piece, and will destroy its own pieces caught in the blast.

Precompute 64 3×3 masks as `u64` bitboards.

### 2.4 Queen — Forced sweeper, maximum 2

- Moves along the standard 8 rays.
- **Any piece she passes over is captured.** She cannot slide past a piece without taking it.
- She may capture a maximum of **2** pieces per move, and **must stop no later than the 2nd**.
- She may stop after 1 capture, or after 0.
- Friendly pieces are captured on the same terms as enemy pieces, up to 2 of them.
- The queen **survives**. No explosion.

Exact enumeration per direction:

```
captures = 0
sq = origin
loop:
    sq += direction
    if sq is off board: break
    if sq is empty:
        emit move(origin -> sq)      # captures the `captures` pieces already passed
        continue
    else:
        captures += 1
        emit move(origin -> sq)      # lands on and captures this piece, plus any passed earlier
        if captures == 2: break
        continue
```

So from a ray with pieces at distances 3 and 6, the legal destinations are: 1, 2 (0 captures), 3 (1 capture), 4, 5 (1 capture), 6 (2 captures). Distance 7+ is unreachable.

### 2.5 King — Standard

- Moves one square in any of 8 directions.
- Captures by displacement only. No explosion, no sweep.
- May capture friendly pieces.
- **No castling.**
- May legally move onto an attacked square.

---

## 3. Terminal conditions

Evaluated **after** the moving side's move fully resolves, in this order:

1. **Both kings removed simultaneously** → **draw**. Possible via a single explosion covering both kings, a knight path crossing both, or a queen sweep taking both.
2. **Exactly one king removed** → the side whose king survives **wins**. Note this includes a player destroying their own king: it is a loss for them regardless of who moved.
3. **Only kings and pawns remain on the board** → **draw**.
4. **Ply cap reached (200 plies)** → see §4.

---

## 4. Ply cap and the shuffle problem

The current rule is a hard cap at 200 plies with no repetition detection.

**Recommended adjudication at the cap: score by material, not an automatic draw.**

Rationale: with no repetition rule, a player who is losing has a trivially optimal strategy — shuffle a piece back and forth until ply 200 and claim the draw. A self-play agent will discover this within a few thousand games and it will corrupt training. Adjudicating by material removes the incentive at zero implementation cost (no Zobrist hashing required, unlike 3-fold repetition).

Suggested adjudication:

```
material_diff = sum(piece_values[own]) - sum(piece_values[opponent])
if material_diff >  threshold: win for that side
if material_diff < -threshold: loss
else: draw
```

with `threshold` around 2.0 using the values in §6.

**Required regardless of choice:** the current ply count must be fed to the evaluation function and to the neural network as an input feature. Without it the value estimate is non-Markovian — a position at ply 20 and the identical position at ply 195 have very different values.

---

## 5. Strategic consequences of the ruleset

These follow directly from the rules and should be reflected in hand-written evaluation.

**5.1. An empty ring is king safety.** A rook or bishop must land on an *occupied* square to detonate. If all 8 neighbours of a king are empty, and no enemy slider can reach the king's own square, that king cannot be exploded. Conversely, **every occupied square adjacent to a king is a live detonator**, including that player's own pieces. This inverts standard chess intuition: pieces clustered near your own king are a liability, not a shield.

**5.2. Pawn teleport is an offensive weapon.** A pawn cannot capture, but it can teleport adjacent to the enemy king to *manufacture* a detonator for a friendly rook or bishop. Expect this to be a primary tactical motif.

**5.3. Rooks and bishops are consumable.** Because they die on capture, their standing value is mostly line control, blocking, and the threat of detonation. Once spent, they are gone. Trading a rook to remove one enemy pawn is close to a pure loss.

**5.4. The knight is the premium piece.** Up to 3 captures, unblockable, and it survives. It also threatens the enemy king along its entire path rather than only at its destination, giving it far more king-attacking squares than any other piece.

**5.5. The trailing side wants simplification.** The "only kings and pawns" draw is the losing player's main resource. A material-down side should be trading pieces off aggressively, including with suicidal rook and bishop detonations.

**5.6. Exchange evaluation is exact, not static.** Standard SEE is meaningless here because there is no recapture sequence to model. The material swing of any capture — a blast, a knight sweep, a queen sweep — is fully computable in one step by intersecting the affected mask with the occupancy bitboards. Use this exact delta for move ordering.

---

## 6. Starting evaluation weights

Guesses for a first alpha-beta pass. Tune by self-play afterwards.

| Piece | Value | Notes |
|---|---|---|
| King | ±∞ | Terminal |
| Queen | 10 | Survives, 2 captures, long range |
| Knight | 9 | Survives, 3 captures, unblockable |
| Rook | 4.5 | Consumable; line control + threat |
| Bishop | 4 | Consumable; can only *land* on its own colour |
| Pawn | 0.25 | Blocker, detonator primer, draw resource |

Positional terms worth adding:

- `−3.0` per occupied square adjacent to own king that an enemy rook or bishop can reach this move (an immediate loss threat)
- `−0.4` per occupied square adjacent to own king generally (latent detonator)
- `+0.3` per enemy piece clustered such that one of our detonations nets ≥ 2 pieces
- Scale the material advantage down as total non-king non-pawn material falls, reflecting the drift toward the kings-and-pawns draw

---

## 7. Test suite

Hand-verify each of these before writing any training code. Rule bugs here will train a confident engine that plays a subtly different game from the opponent's engine.

**Explosions**
- [ ] Rook captures on a corner square — 2×2 clipped region cleared, mover removed
- [ ] Rook capture that destroys the moving side's own queen standing adjacent to the target
- [ ] Bishop capture whose blast contains both kings → draw, not a win
- [ ] Bishop capture whose blast contains only the mover's own king → the mover loses
- [ ] Rook cannot slide past a blocking pawn to reach a target behind it

**Knight sweeps**
- [ ] Knight `(+2,+1)` over two friendly pieces — both removed, knight survives on destination
- [ ] Knight passes over the enemy king at an intermediate square and lands on an empty square → win
- [ ] Knight path is never blocked; a fully occupied path is still a legal move

**Queen sweeps**
- [ ] Ray with pieces at distance 3 and 6 → exactly the destination set {1,2,3,4,5,6}
- [ ] Queen may not reach distance 7 past two pieces
- [ ] Queen capturing two of her own pieces is legal
- [ ] Queen capturing one enemy piece and stopping on the next empty square

**Pawns**
- [ ] Pawn cannot move to an occupied square, including an undefended enemy piece
- [ ] Pawn on the 8th rank does not promote and may still teleport
- [ ] Pawn correctly blocks a rook ray
- [ ] Pawn teleport count equals `pawns × empty_squares`

**Terminal**
- [ ] Simultaneous double king removal → draw in all three ways it can occur (explosion, knight, queen)
- [ ] Board reduced to kings + pawns → draw, detected immediately
- [ ] Board reduced to two bare kings → draw
- [ ] Ply cap reached → adjudicated per §4

**Perft-style**
- [ ] Node counts at depths 1–4 from the start position match between your engine and a slow, obviously-correct reference implementation written independently
