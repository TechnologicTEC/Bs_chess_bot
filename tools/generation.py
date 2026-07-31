#!/usr/bin/env python3
"""Run one generation of the training loop (build plan, Phase 5).

    python tools/generation.py --gen 1 --games 20000 --depth 6
    python tools/generation.py --gen 2 --games 20000 --depth 6      # uses gen 1's net

Each generation is:

    self-play M games at fixed depth, parallel across all cores, randomised
    openings, keeping only positions whose best move is quiet
      -> train the net in PyTorch
      -> verify the exported weights evaluate identically in both languages
      -> gate: new engine vs previous, promote above the threshold

Generation 0 is the hand evaluation; there is nothing to train. Most of the
strength gain arrives in generations 1-4 and returns flatten after that.

Nothing here is destructive: each generation writes its own data shard and net,
and a failed gate leaves the previous generation in place as `nets/current.bin`.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def binary(name):
    for candidate in (name + ".exe", name):
        path = os.path.join(ROOT, "target", "release", candidate)
        if os.path.exists(path):
            return path
    sys.exit("%s not built — run `cargo build --release` first" % name)


def step(title, cmd, check=True):
    print("\n=== %s ===" % title)
    print("    %s" % " ".join(str(c) for c in cmd))
    t0 = time.time()
    proc = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    print("    [%s in %.1fs]" % ("ok" if proc.returncode == 0 else
                                 "exit %d" % proc.returncode, time.time() - t0))
    if check and proc.returncode != 0:
        sys.exit("step failed: %s" % title)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen", type=int, required=True, help="generation to produce (>= 1)")
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gate-pairs", type=int, default=100)
    ap.add_argument("--gate", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--net-dir", default="nets")
    ap.add_argument("--keep-going", action="store_true",
                    help="do not stop when the gate fails")
    args = ap.parse_args()

    if args.gen < 1:
        sys.exit("generation 0 is the hand evaluation — there is nothing to train")

    os.makedirs(os.path.join(ROOT, args.data_dir), exist_ok=True)
    os.makedirs(os.path.join(ROOT, args.net_dir), exist_ok=True)

    previous = os.path.join(args.net_dir, "gen%d.bin" % (args.gen - 1))
    candidate = os.path.join(args.net_dir, "gen%d.bin" % args.gen)
    shard = os.path.join(args.data_dir, "gen%d.bin" % args.gen)
    seed = args.seed if args.seed is not None else args.gen * 1_000_003

    has_previous = args.gen > 1 and os.path.exists(os.path.join(ROOT, previous))
    if args.gen > 1 and not has_previous:
        sys.exit("%s does not exist — run generation %d first" % (previous, args.gen - 1))

    print("generation %d" % args.gen)
    print("  data      %s" % shard)
    print("  candidate %s" % candidate)
    print("  opponent  %s" % (previous if has_previous else "hand evaluation"))

    # 1. Self-play with the previous generation's evaluator.
    selfplay = [binary("selfplay"), "--games", args.games, "--depth", args.depth,
                "--out", shard, "--seed", seed]
    if has_previous:
        selfplay += ["--net", previous]
    step("self-play", selfplay)

    # 2. Train.
    step("train", [sys.executable, os.path.join("tools", "train.py"),
                   "--data", shard, "--out", candidate,
                   "--epochs", args.epochs, "--batch", args.batch, "--lr", args.lr])

    # 3. Verify the weights mean the same thing in both languages.
    step("verify weights", [sys.executable, os.path.join("tools", "verify_net.py"),
                            "--net", candidate])

    # 4. Gate against the previous generation (or the hand eval for generation 1).
    gate_cmd = [binary("gauntlet"), "--pairs", args.gate_pairs, "--depth", args.depth,
                "--a", candidate, "--b", previous if has_previous else "hand",
                "--gate", args.gate, "--seed", seed ^ 0xABCDEF]
    passed = step("gate", gate_cmd, check=False) == 0

    current = os.path.join(ROOT, args.net_dir, "current.bin")
    if passed:
        shutil.copyfile(os.path.join(ROOT, candidate), current)
        print("\nPROMOTED — %s is now %s/current.bin" % (candidate, args.net_dir))
    else:
        print("\nHELD — %s did not clear the gate; current.bin unchanged" % candidate)
        print("Options: more games, more epochs, or a lower --lr.")
        if not args.keep_going:
            sys.exit(1)

    print("\nnext: python tools/generation.py --gen %d --games %d --depth %d"
          % (args.gen + 1, args.games, args.depth))


if __name__ == "__main__":
    main()
