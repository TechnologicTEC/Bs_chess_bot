#!/usr/bin/env python3
"""Check that an exported network evaluates identically in PyTorch and in Rust.

    python tools/verify_net.py --net nets/gen1.bin

The weight-serialisation seam is the one place in this project where a bug is
completely silent: transpose a matrix and nothing crashes, nothing fails a test,
the engine just plays badly and you spend a week blaming the training data. So it
gets its own check.

Loads the .bin into the PyTorch model, evaluates a spread of positions, asks the
Rust engine for the same numbers, and diffs them.
"""

import argparse
import os
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import HL, L1, L2, MAGIC, NUM_FEATURES, VERSION, Nnue  # noqa: E402

try:
    import torch
except ImportError:
    sys.exit("PyTorch is required: pip install torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b - - 1 1",
    "4k3/8/8/3n4/8/8/8/R3K3 w - - 0 1",
    "4k3/8/8/3n4/8/8/8/R3K3 b - - 1 1",
    "k7/8/8/8/8/8/8/K1Q5 w - - 0 1",
    "7k/8/8/8/8/8/8/K6R b - - 9 5",
    "4k2r/8/8/8/2Qp4/8/8/3RK3 w - - 12 7",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w - - 30 16",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 44 23",
    "3qk3/8/8/8/8/8/8/3QK3 b - - 3 2",
]


def load_into_model(path):
    with open(path, "rb") as fh:
        blob = fh.read()
    magic, version, nf, hl, l1, l2 = struct.unpack_from("<6I", blob, 0)
    if magic != MAGIC:
        sys.exit("%s is not a bschess network file" % path)
    if version != VERSION:
        sys.exit("unsupported network version %d" % version)
    if (nf, l1, l2) != (NUM_FEATURES, L1, L2):
        sys.exit("network shape %dx%dx%dx%d does not match the trainer" % (nf, hl, l1, l2))

    floats = np.frombuffer(blob, dtype="<f4", offset=24)
    model = Nnue(hl)
    HL = hl
    cursor = 0

    def take(shape):
        nonlocal cursor
        n = int(np.prod(shape))
        chunk = floats[cursor:cursor + n]
        if chunk.size != n:
            sys.exit("network file truncated")
        cursor += n
        return torch.from_numpy(chunk.reshape(shape).copy())

    with torch.no_grad():
        # ft is stored feature-major; nn.Linear wants (out, in).
        model.ft.weight.copy_(take((NUM_FEATURES, HL)).t())
        model.ft.bias.copy_(take((HL,)))
        model.l1.weight.copy_(take((L1, 2 * HL)))
        model.l1.bias.copy_(take((L1,)))
        model.l2.weight.copy_(take((L2, L1)))
        model.l2.bias.copy_(take((L2,)))
        model.l3.weight.copy_(take((1, L2)))
        model.l3.bias.copy_(take((1,)))

    if cursor != floats.size:
        sys.exit("network file has %d trailing floats" % (floats.size - cursor))
    model.eval()
    # The exported final layer has the centipawn scale folded in, so this
    # reloaded model returns centipawns rather than the logit `Nnue.forward`
    # produces during training. That is exactly what we want to compare.
    return model


def features_from_fen(fen):
    """Independent FEN reader — deliberately not shared with the engine."""
    board = {}
    placement, side = fen.split()[0], fen.split()[1]
    rank, file = 7, 0
    for ch in placement:
        if ch == "/":
            rank, file = rank - 1, 0
        elif ch.isdigit():
            file += int(ch)
        else:
            colour = 0 if ch.isupper() else 1
            ptype = "PNBRQK".index(ch.upper())
            board[rank * 8 + file] = (colour, ptype)
            file += 1
    stm = 0 if side == "w" else 1

    own, opp = [], []
    for square, (colour, ptype) in board.items():
        own.append(((1 if colour != stm else 0) * 6 + ptype) * 64 + square)
        opp.append(((1 if colour == stm else 0) * 6 + ptype) * 64 + square)
    return own, opp


def torch_eval(model, fens):
    n, width = len(fens), 32
    own = torch.zeros((n, width), dtype=torch.int64)
    opp = torch.zeros((n, width), dtype=torch.int64)
    mask = torch.zeros((n, width, 1), dtype=torch.float32)
    for i, fen in enumerate(fens):
        o, p = features_from_fen(fen)
        own[i, :len(o)] = torch.tensor(o, dtype=torch.int64)
        opp[i, :len(p)] = torch.tensor(p, dtype=torch.int64)
        mask[i, :len(o), 0] = 1.0
    with torch.no_grad():
        return model(own, opp, mask).numpy()


def engine_binary():
    for candidate in ("bschess.exe", "bschess"):
        path = os.path.join(ROOT, "target", "release", candidate)
        if os.path.exists(path):
            return path
    sys.exit("engine not built — run `cargo build --release` first")


def rust_eval(net_path, fens):
    proc = subprocess.run(
        [engine_binary(), "--eval-fens", net_path],
        input="\n".join(fens) + "\n",
        capture_output=True, text=True, cwd=ROOT,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit("engine failed")
    out = {}
    for line in proc.stdout.splitlines():
        if "\t" in line:
            fen, value = line.rsplit("\t", 1)
            out[fen] = int(value)
    return [out[f] for f in fens]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", required=True)
    ap.add_argument("--tolerance", type=float, default=1.5,
                    help="allowed centipawn difference (Rust truncates to int)")
    args = ap.parse_args()

    model = load_into_model(args.net)
    theirs = rust_eval(args.net, POSITIONS)
    ours = torch_eval(model, POSITIONS)

    worst = 0.0
    print("%-60s %10s %10s %8s" % ("position", "pytorch", "rust", "delta"))
    for fen, t, r in zip(POSITIONS, ours, theirs):
        delta = abs(float(t) - r)
        worst = max(worst, delta)
        flag = "" if delta <= args.tolerance else "   <-- MISMATCH"
        print("%-60s %10.2f %10d %8.2f%s" % (fen[:60], t, r, delta, flag))

    if worst <= args.tolerance:
        print("\nOK — largest difference %.2f cp, within tolerance %.2f"
              % (worst, args.tolerance))
        return 0
    print("\nMISMATCH — largest difference %.2f cp exceeds tolerance %.2f"
          % (worst, args.tolerance))
    print("The exported weights do not mean the same thing in both places.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
