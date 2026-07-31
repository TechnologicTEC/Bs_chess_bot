#!/usr/bin/env python3
"""Train an NNUE for Bullshit Chess (build plan, Phase 5).

    python tools/train.py --data data/gen1.bin --out nets/gen1.bin --epochs 20

Reads the fixed-width 32-byte records written by `selfplay`, trains the
768-256x2-32-32-1 network, and writes weights in the binary format
`src/nnue.rs` loads.

Three things here are variant-specific and matter:

* **The quiet filter.** Standard NNUE practice trains evaluation only on
  positions where nothing is hanging, because quiescence handles tactics. In a
  variant this volatile, training on tactical positions teaches the net noise.
  Self-play tags each record; this script keeps only tagged ones by default.

* **8x augmentation is free and correct.** No castling and no directional pawn
  movement means the rules are fully invariant under the dihedral group of the
  square, so every position can be rotated and mirrored eight ways with the same
  label. That is a genuine 8x on data volume, not a regulariser hack.

  It is applied *on the fly*, one random symmetry per sample per epoch, rather
  than by materialising 8 copies. Materialising costs ~20 GB on a realistic
  5M-position generation, and resampling each epoch is strictly better anyway.

* **No ply-count input.** The ply cap is handled by the search adjudicating at
  the horizon, so the network never needs to know the ply.

The network outputs centipawns directly, with the loss taken in win-probability
space — the standard NNUE recipe. That keeps the output in the units the search
already speaks, so nothing needs rescaling on the Rust side.
"""

import argparse
import os
import struct
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("PyTorch is required: pip install torch")

RECORD_SIZE = 32
MAX_PIECES = 32
NUM_FEATURES = 768
HL = 256
L1 = 32
L2 = 32
MAGIC = 0x4E4E5342  # "BSNN"
VERSION = 1
FLAG_QUIET = 1


# ---------------------------------------------------------------------------
# Dihedral symmetry — must agree with src/symmetry.rs
# ---------------------------------------------------------------------------

def build_square_permutations():
    """perm[i][s] is where square s lands under symmetry i. Square index is
    rank * 8 + file, so a1 = 0 and h8 = 63."""
    def apply(i, r, c):
        return [
            (r, c),          # identity
            (c, 7 - r),      # rotate 90
            (7 - r, 7 - c),  # rotate 180
            (7 - c, r),      # rotate 270
            (r, 7 - c),      # mirror across the vertical axis
            (7 - r, c),      # mirror across the horizontal axis
            (c, r),          # transpose
            (7 - c, 7 - r),  # anti-transpose
        ][i]

    perms = np.zeros((8, 64), dtype=np.int64)
    for i in range(8):
        for s in range(64):
            r, c = apply(i, s // 8, s % 8)
            perms[i, s] = r * 8 + c
    return perms


SQUARE_PERM = build_square_permutations()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(paths, quiet_only=True):
    """Unpack the binary shards into dense per-piece arrays.

    Returns (pieces, squares, stm, score, result) where `pieces` and `squares`
    are (N, 32) int8/int16 with -1 padding in unused slots.
    """
    chunks = []
    for path in paths:
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size == 0:
            sys.exit("%s is empty" % path)
        if raw.size % RECORD_SIZE:
            sys.exit("%s is not a whole number of %d-byte records" % (path, RECORD_SIZE))
        chunks.append(raw.reshape(-1, RECORD_SIZE))
        print("  %-40s %9d records" % (path, chunks[-1].shape[0]))
    rec = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

    if quiet_only:
        keep = (rec[:, 29] & FLAG_QUIET) != 0
        print("  quiet filter: keeping %d of %d (%.1f%%)"
              % (keep.sum(), keep.size, 100.0 * keep.sum() / max(keep.size, 1)))
        rec = rec[keep]
    if rec.shape[0] == 0:
        sys.exit("no records survived the quiet filter")

    n = rec.shape[0]
    stm = np.ascontiguousarray(rec[:, 24]).astype(np.int8)
    score = np.ascontiguousarray(rec[:, 26:28]).view(np.int16).reshape(-1).astype(np.float32)
    result = np.ascontiguousarray(rec[:, 28]).view(np.int8).astype(np.float32)

    # Column s of `bits` is 1 when square s is occupied. Little bit order matches
    # the little-endian u64 the engine writes.
    bits = np.unpackbits(np.ascontiguousarray(rec[:, 0:8]), axis=1, bitorder="little")
    counts = bits.sum(axis=1).astype(np.int64)
    if counts.max() > MAX_PIECES:
        sys.exit("a record claims more than %d pieces" % MAX_PIECES)

    # Occupied square indices, ascending. Stable argsort of the negated mask puts
    # the set bits first while preserving their natural order.
    order = np.argsort(-bits, axis=1, kind="stable")[:, :MAX_PIECES].astype(np.int16)

    # Nibble j of the record is the piece on the j-th occupied square, ascending.
    nibbles = rec[:, 8:24]
    pieces = np.empty((n, MAX_PIECES), dtype=np.int8)
    pieces[:, 0::2] = nibbles & 0x0F
    pieces[:, 1::2] = nibbles >> 4

    valid = np.arange(MAX_PIECES)[None, :] < counts[:, None]
    squares = np.where(valid, order, -1).astype(np.int16)
    pieces = np.where(valid, pieces, -1).astype(np.int8)

    return pieces, squares, stm, score, result, counts


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Nnue(nn.Module):
    """768 -> 256 per perspective (weights shared) -> concat 512 -> 32 -> 32 -> 1.

    ClippedReLU throughout, matching src/nnue.rs. ~200k parameters.

    `forward` returns a **win-probability logit**, not centipawns. Training the
    output directly in centipawns puts every gradient through a 1/400 divisor and
    the thing simply never moves off zero. The centipawn scale is folded into the
    final layer at export time instead, so the engine still reads centipawns and
    src/nnue.rs needs no knowledge of any of this.
    """

    def __init__(self):
        super().__init__()
        self.ft = nn.Linear(NUM_FEATURES, HL)
        self.l1 = nn.Linear(2 * HL, L1)
        self.l2 = nn.Linear(L1, L2)
        self.l3 = nn.Linear(L2, 1)
        # Only ~32 of the 768 features are ever active, so the accumulator needs a
        # wider init than fan-in would suggest to land inside ClippedReLU's live
        # range rather than pinned at 0.
        nn.init.normal_(self.ft.weight, std=0.06)
        nn.init.constant_(self.ft.bias, 0.5)
        # Zero-init the output so epoch 1 predicts "equal" rather than noise. The
        # layer below gets no gradient on the very first step and every step after.
        nn.init.zeros_(self.l3.weight)
        nn.init.zeros_(self.l3.bias)

    @staticmethod
    def crelu(x):
        return torch.clamp(x, 0.0, 1.0)

    def accumulate(self, indices, mask):
        """indices: (B, 32) int64, mask: (B, 32, 1) float. Sparse sum of feature
        columns plus the bias — the same full refresh the engine does per node."""
        cols = self.ft.weight.t()[indices]            # (B, 32, HL)
        return (cols * mask).sum(dim=1) + self.ft.bias

    def forward(self, own, opp, mask):
        x = torch.cat(
            [self.crelu(self.accumulate(own, mask)),
             self.crelu(self.accumulate(opp, mask))],
            dim=1,
        )
        x = self.crelu(self.l1(x))
        x = self.crelu(self.l2(x))
        return self.l3(x).squeeze(-1)


def export(model, path, scale):
    """Write weights in the layout src/nnue.rs reads: header, then ft_weight
    (row-major by feature), ft_bias, w1, b1, w2, b2, w3, b3, all f32 LE.

    `scale` (centipawns per logit) is folded into the final layer here, so the
    engine reads centipawns straight out of the network.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<6I", MAGIC, VERSION, NUM_FEATURES, HL, L1, L2))
        # PyTorch stores Linear weight as (out, in). The engine indexes the
        # feature transformer by feature, so it goes out transposed.
        for tensor in (
            model.ft.weight.t().contiguous(),
            model.ft.bias,
            model.l1.weight.contiguous(),
            model.l1.bias,
            model.l2.weight.contiguous(),
            model.l2.bias,
            model.l3.weight.contiguous() * scale,
            model.l3.bias * scale,
        ):
            fh.write(tensor.detach().cpu().numpy().astype("<f4").tobytes(order="C"))


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------

def make_batch(pieces, squares, stm, perm_table, idx, device, symmetries, generator):
    """Build one batch's feature indices, applying a random dihedral symmetry per
    sample. Piece identity, side to move and the label are all invariant under
    the group; only the squares move.

    A feature is `(relative_colour * 6 + piece_type) * 64 + square`, where the
    relative colour bit is the whole canonicalisation — there is no board flip,
    because no piece in this variant has a forward direction.
    """
    p = pieces[idx].to(device, non_blocking=True).long()      # (B, 32), -1 padded
    s = squares[idx].to(device, non_blocking=True).long()
    side = stm[idx].to(device, non_blocking=True).long().unsqueeze(1)  # (B, 1)

    valid = p >= 0
    mask = valid.unsqueeze(-1).float()

    if symmetries > 1:
        which = torch.randint(0, symmetries, (p.shape[0], 1), device=device,
                              generator=generator)
        s = torch.gather(perm_table[which.squeeze(1)], 1, s.clamp(min=0))
    else:
        s = s.clamp(min=0)

    colour = torch.div(p.clamp(min=0), 6, rounding_mode="floor")
    ptype = p.clamp(min=0) % 6

    own = ((colour != side).long() * 6 + ptype) * 64 + s
    opp = ((colour == side).long() * 6 + ptype) * 64 + s
    # Padding slots point at feature 0 and are zeroed by the mask.
    own = torch.where(valid, own, torch.zeros_like(own))
    opp = torch.where(valid, opp, torch.zeros_like(opp))
    return own, opp, mask


# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+", required=True, help="one or more .bin shards")
    ap.add_argument("--out", default="nets/gen1.bin")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--augment", type=int, default=8, choices=range(1, 9),
                    help="how many of the 8 dihedral symmetries to sample from")
    ap.add_argument("--lambda-result", type=float, default=0.3,
                    help="label = (1-l) * search_score + l * game_result")
    ap.add_argument("--scale", type=float, default=400.0,
                    help="centipawns per logit when mapping to win probability")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--all-positions", action="store_true",
                    help="disable the quiet filter (not recommended)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("loading:")
    pieces, squares, stm, score, result, counts = load_records(
        args.data, quiet_only=not args.all_positions
    )
    n = pieces.shape[0]
    print("  %d positions, %.1f pieces on average" % (n, counts.mean()))
    print("  score  mean %+.0f cp, sd %.0f cp" % (score.mean(), score.std()))
    print("  result mean %+.3f (from White's view)" % result.mean())

    # Labels in win-probability space. The search score is side-to-move relative;
    # the game result is stored from White's view, so flip it when Black moves.
    result_stm = np.where(stm == 1, -result, result)
    target = (
        (1.0 - args.lambda_result) * sigmoid(score / args.scale)
        + args.lambda_result * (result_stm + 1.0) / 2.0
    ).astype(np.float32)

    order = np.random.permutation(n)
    pieces, squares, stm, target = (
        pieces[order], squares[order], stm[order], target[order]
    )

    n_val = max(1, min(int(n * args.val_fraction), 200_000))
    n_train = n - n_val
    if n_train <= 0:
        sys.exit("not enough data to train on")
    print("  %d train / %d validation, %dx augmentation sampled per epoch"
          % (n_train, n_val, args.augment))

    device = torch.device(args.device)
    print("device: %s" % device)

    pieces_t = torch.from_numpy(pieces)
    squares_t = torch.from_numpy(squares)
    stm_t = torch.from_numpy(stm)
    target_t = torch.from_numpy(target)
    perm_table = torch.from_numpy(SQUARE_PERM).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    model = Nnue().to(device)
    print("model: %d parameters" % sum(p.numel() for p in model.parameters()))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    loss_fn = nn.MSELoss()

    val_idx = torch.arange(n_val)
    steps = max(1, n_train // args.batch)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n_train) + n_val
        running = 0.0
        for step in range(steps):
            idx = perm[step * args.batch:(step + 1) * args.batch]
            own, opp, mask = make_batch(
                pieces_t, squares_t, stm_t, perm_table, idx, device,
                args.augment, generator,
            )
            y = target_t[idx].to(device, non_blocking=True)

            # The model emits a logit; the loss lives in win-probability space.
            loss = loss_fn(torch.sigmoid(model(own, opp, mask)), y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
        sched.step()

        # Validate on the identity symmetry only, so the number is comparable
        # across epochs.
        model.eval()
        with torch.no_grad():
            vloss, vsum, vn = 0.0, 0.0, 0
            preds = []
            for start in range(0, n_val, args.batch):
                idx = val_idx[start:start + args.batch]
                own, opp, mask = make_batch(
                    pieces_t, squares_t, stm_t, perm_table, idx, device, 1, generator
                )
                y = target_t[idx].to(device)
                pred = model(own, opp, mask)
                vsum += loss_fn(torch.sigmoid(pred), y).item() * idx.numel()
                vn += idx.numel()
                preds.append(pred)
            vloss = vsum / max(vn, 1)
            # Reported in the units the engine will see.
            spread = torch.cat(preds).std().item() * args.scale

        marker = ""
        if vloss < best_val:
            best_val = vloss
            export(model, args.out, args.scale)
            marker = "  <- saved"
        print("epoch %3d  train %.5f  val %.5f  output sd %6.1f cp  %5.1fs%s"
              % (epoch, running / steps, vloss, spread, time.time() - t0, marker))

    print("\nbest validation loss %.5f -> %s (%d bytes)"
          % (best_val, args.out, os.path.getsize(args.out)))
    print("\nnow gate it before promoting:")
    print("  cargo run --release --bin gauntlet -- --pairs 100 --depth 6 "
          "--a %s --b <previous generation>" % args.out)


if __name__ == "__main__":
    main()
