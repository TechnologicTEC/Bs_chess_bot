#!/usr/bin/env python3
"""The Phase 1 gate: run the Rust engine and the Python reference over the same
suite files and diff them.

    python tools/perft_gate.py                     # curated + random suites
    python tools/perft_gate.py --suite tests/perft_suite.txt

A movegen bug found after two weeks of training is two weeks lost, and you cannot
detect one by watching games. Do not proceed past a mismatch.

Exit code 0 means every node count agreed.
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SUITES = [
    os.path.join("tests", "perft_suite.txt"),
    os.path.join("tests", "perft_random_suite.txt"),
]


def engine_binary():
    for candidate in ("perft.exe", "perft"):
        path = os.path.join(ROOT, "target", "release", candidate)
        if os.path.exists(path):
            return path
    sys.exit("engine not built — run `cargo build --release` first")


def run(cmd):
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit("command failed: %s" % " ".join(cmd))
    return [line for line in proc.stdout.splitlines() if line.strip()]


def check(suite):
    print("=== %s ===" % suite)
    rust = run([engine_binary(), "--suite", suite])
    ref = run([sys.executable, os.path.join("ref", "reference.py"), "--suite", suite])

    if len(rust) != len(ref):
        print("  line count differs: engine %d, reference %d" % (len(rust), len(ref)))
        return False

    mismatches = [(a, b) for a, b in zip(rust, ref) if a != b]
    total = 0
    for line in ref:
        try:
            total += int(line.split("\t")[-1])
        except ValueError:
            pass

    if mismatches:
        print("  MISMATCH in %d of %d counts:" % (len(mismatches), len(rust)))
        for a, b in mismatches[:20]:
            print("    engine    %s" % a)
            print("    reference %s" % b)
        return False

    print("  OK — %d counts agree, %d nodes checked" % (len(rust), total))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", action="append", default=None)
    args = ap.parse_args()

    suites = args.suite or DEFAULT_SUITES
    ok = all([check(s) for s in suites])

    if ok:
        print("\nGATE PASSED")
    else:
        print("\nGATE FAILED — do not proceed to training")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
