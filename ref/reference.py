#!/usr/bin/env python3
"""Deliberately slow, obviously-correct reference implementation of Bullshit Chess.

This exists for exactly one purpose: to disagree with the Rust engine if the Rust
engine is wrong. So it is written to be readable against `bullshit-chess-spec.md`
line by line, in a completely different style from the engine — a mailbox board of
(rank, file) tuples, no bitboards, no precomputed tables, no cleverness anywhere.

A movegen bug found after two weeks of training is two weeks lost, and you cannot
detect one by watching games.

Usage:
    python ref/reference.py --depth 3
    python ref/reference.py --fen "<fen>" --depth 4 --divide
    python ref/reference.py --suite tests/perft_suite.txt

Perft is defined as (and must match src/perft.rs exactly):

    perft(pos, 0) = 1
    perft(pos, d) = 0                              if pos is terminal (spec section 3)
    perft(pos, d) = sum(perft(child, d-1) for each legal move)
"""

import argparse
import sys
import time

WHITE, BLACK = "w", "b"
PLY_CAP = 200
ADJUDICATION_THRESHOLD = 2.0

# Spec section 6 starting weights. Used only by the ply-cap adjudication.
PIECE_VALUE = {"P": 0.25, "N": 9.0, "B": 4.0, "R": 4.5, "Q": 10.0, "K": 0.0}

# Spec section 2.2, transcribed straight from the path table:
# (destination, intermediate 1, intermediate 2), each as (d_rank, d_file).
KNIGHT_PATHS = [
    ((+2, +1), (+1, 0), (+2, 0)),
    ((+2, -1), (+1, 0), (+2, 0)),
    ((-2, +1), (-1, 0), (-2, 0)),
    ((-2, -1), (-1, 0), (-2, 0)),
    ((+1, +2), (0, +1), (0, +2)),
    ((-1, +2), (0, +1), (0, +2)),
    ((+1, -2), (0, -1), (0, -2)),
    ((-1, -2), (0, -1), (0, -2)),
]

ROOK_DIRS = [(+1, 0), (0, +1), (-1, 0), (0, -1)]
BISHOP_DIRS = [(+1, +1), (-1, +1), (-1, -1), (+1, -1)]
QUEEN_DIRS = ROOK_DIRS + BISHOP_DIRS
KING_DIRS = QUEEN_DIRS


def on_board(r, c):
    return 0 <= r < 8 and 0 <= c < 8


def square_name(r, c):
    return "abcdefgh"[c] + "12345678"[r]


def parse_square(s):
    return ("12345678".index(s[1]), "abcdefgh".index(s[0]))


class Position:
    """board maps (rank, file) -> (colour, piece letter). Absent key means empty."""

    def __init__(self, board=None, side=WHITE, ply=0):
        self.board = dict(board) if board else {}
        self.side = side
        self.ply = ply

    def copy(self):
        return Position(self.board, self.side, self.ply)

    # -- setup ------------------------------------------------------------

    @staticmethod
    def from_fen(fen):
        parts = fen.split()
        pos = Position()
        rank, file = 7, 0
        for ch in parts[0]:
            if ch == "/":
                rank, file = rank - 1, 0
            elif ch.isdigit():
                file += int(ch)
            else:
                colour = WHITE if ch.isupper() else BLACK
                pos.board[(rank, file)] = (colour, ch.upper())
                file += 1
        pos.side = WHITE if len(parts) < 2 or parts[1] == "w" else BLACK
        pos.ply = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0
        return pos

    @staticmethod
    def startpos():
        return Position.from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
        )

    def to_fen(self):
        rows = []
        for rank in range(7, -1, -1):
            row, run = "", 0
            for file in range(8):
                p = self.board.get((rank, file))
                if p is None:
                    run += 1
                else:
                    if run:
                        row += str(run)
                        run = 0
                    row += p[1] if p[0] == WHITE else p[1].lower()
            if run:
                row += str(run)
            rows.append(row)
        return "%s %s - - %d %d" % (
            "/".join(rows),
            self.side,
            self.ply,
            self.ply // 2 + 1,
        )

    def __str__(self):
        out = []
        for rank in range(7, -1, -1):
            row = []
            for file in range(8):
                p = self.board.get((rank, file))
                row.append("." if p is None else (p[1] if p[0] == WHITE else p[1].lower()))
            out.append("%d | %s" % (rank + 1, " ".join(row)))
        out.append("    a b c d e f g h")
        return "\n".join(out)

    # -- move generation, spec section 2 ----------------------------------

    def moves(self):
        out = []
        for (r, c), (colour, piece) in sorted(self.board.items()):
            if colour != self.side:
                continue
            if piece == "P":
                out.extend(self._pawn_moves(r, c))
            elif piece == "N":
                out.extend(self._knight_moves(r, c))
            elif piece == "R":
                out.extend(self._slider_moves(r, c, ROOK_DIRS))
            elif piece == "B":
                out.extend(self._slider_moves(r, c, BISHOP_DIRS))
            elif piece == "Q":
                out.extend(self._queen_moves(r, c))
            elif piece == "K":
                out.extend(self._king_moves(r, c))
        return out

    def _pawn_moves(self, r, c):
        # 2.1: any empty square on the board, regardless of distance or direction.
        # Cannot capture, so occupied squares are simply excluded, and the pawn's
        # own square is occupied, so there is no null move.
        return [
            ((r, c), (tr, tc))
            for tr in range(8)
            for tc in range(8)
            if (tr, tc) not in self.board
        ]

    def _knight_moves(self, r, c):
        # 2.2: legal iff the destination is on the board. Never blocked.
        out = []
        for (dr, dc), _, _ in KNIGHT_PATHS:
            if on_board(r + dr, c + dc):
                out.append(((r, c), (r + dr, c + dc)))
        return out

    def _slider_moves(self, r, c, dirs):
        # 2.3: slide along the ray, stopping ON the first occupied square (landing
        # there is the detonating move) and never past it.
        out = []
        for dr, dc in dirs:
            tr, tc = r + dr, c + dc
            while on_board(tr, tc):
                out.append(((r, c), (tr, tc)))
                if (tr, tc) in self.board:
                    break
                tr, tc = tr + dr, tc + dc
        return out

    def _queen_moves(self, r, c):
        # 2.4: transcription of the spec's enumeration pseudocode.
        out = []
        for dr, dc in QUEEN_DIRS:
            captures = 0
            tr, tc = r + dr, c + dc
            while on_board(tr, tc):
                out.append(((r, c), (tr, tc)))
                if (tr, tc) in self.board:
                    captures += 1
                    if captures == 2:
                        break
                tr, tc = tr + dr, tc + dc
        return out

    def _king_moves(self, r, c):
        # 2.5: one square, any direction, may capture friendlies.
        return [
            ((r, c), (r + dr, c + dc))
            for dr, dc in KING_DIRS
            if on_board(r + dr, c + dc)
        ]

    # -- move application --------------------------------------------------

    def make(self, move):
        (fr, fc), (tr, tc) = move
        colour, piece = self.board[(fr, fc)]
        new = self.copy()

        if piece == "P":
            assert (tr, tc) not in self.board, "pawn cannot move onto an occupied square"
            del new.board[(fr, fc)]
            new.board[(tr, tc)] = (colour, piece)

        elif piece == "N":
            # Captures every piece on all three path squares; the knight survives.
            path = None
            for (dr, dc), (i1r, i1c), (i2r, i2c) in KNIGHT_PATHS:
                if (fr + dr, fc + dc) == (tr, tc):
                    path = [(fr + i1r, fc + i1c), (fr + i2r, fc + i2c), (tr, tc)]
                    break
            assert path is not None, "not a knight move"
            del new.board[(fr, fc)]
            for square in path:
                new.board.pop(square, None)
            new.board[(tr, tc)] = (colour, piece)

        elif piece in ("R", "B"):
            if (tr, tc) in self.board:
                # Detonation: clear the 3x3 centred on the destination, clipped at
                # the edges, and remove the mover too. No chain reactions.
                del new.board[(fr, fc)]
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if on_board(tr + dr, tc + dc):
                            new.board.pop((tr + dr, tc + dc), None)
            else:
                del new.board[(fr, fc)]
                new.board[(tr, tc)] = (colour, piece)

        elif piece == "Q":
            # Everything passed over is captured, plus the landing square. The
            # queen survives.
            dr = (tr > fr) - (tr < fr)
            dc = (tc > fc) - (tc < fc)
            del new.board[(fr, fc)]
            sr, sc = fr + dr, fc + dc
            while (sr, sc) != (tr, tc):
                new.board.pop((sr, sc), None)
                sr, sc = sr + dr, sc + dc
            new.board.pop((tr, tc), None)
            new.board[(tr, tc)] = (colour, piece)

        elif piece == "K":
            del new.board[(fr, fc)]
            new.board.pop((tr, tc), None)
            new.board[(tr, tc)] = (colour, piece)

        new.side = BLACK if self.side == WHITE else WHITE
        new.ply = self.ply + 1
        return new

    # -- terminal conditions, spec section 3, in order ---------------------

    def result(self):
        kings = {colour for colour, piece in self.board.values() if piece == "K"}
        white_king, black_king = WHITE in kings, BLACK in kings

        if not white_king and not black_king:
            return "draw"
        if not white_king:
            return "black"
        if not black_king:
            return "white"
        if all(piece in ("K", "P") for _, piece in self.board.values()):
            return "draw"
        if self.ply >= PLY_CAP:
            return self.adjudicate()
        return None

    def adjudicate(self):
        diff = 0.0
        for colour, piece in self.board.values():
            v = PIECE_VALUE[piece]
            diff += v if colour == WHITE else -v
        if diff > ADJUDICATION_THRESHOLD:
            return "white"
        if diff < -ADJUDICATION_THRESHOLD:
            return "black"
        return "draw"


def perft(pos, depth):
    if depth == 0:
        return 1
    if pos.result() is not None:
        return 0
    if depth == 1:
        return len(pos.moves())
    return sum(perft(pos.make(m), depth - 1) for m in pos.moves())


def move_name(move):
    (fr, fc), (tr, tc) = move
    return square_name(fr, fc) + square_name(tr, tc)


def run_divide(pos, depth):
    rows = []
    for m in pos.moves():
        n = 1 if depth <= 1 else perft(pos.make(m), depth - 1)
        rows.append((move_name(m), n))
    for name, n in sorted(rows):
        print("%s\t%d" % (name, n))
    print("# total\t%d" % sum(n for _, n in rows), file=sys.stderr)


def run_suite(path):
    """Suite file lines: depth<TAB>fen<TAB>label. Both engines read the same file,
    so neither can drift onto a different set of positions."""
    total_start = time.time()
    # utf-8-sig: suite files generated by redirecting Windows console output pick
    # up a BOM, which would otherwise land in the first depth field.
    with open(path, encoding="utf-8-sig") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            depth_s, fen, label = line.split("\t")
            pos = Position.from_fen(fen)
            for d in range(1, int(depth_s) + 1):
                t = time.time()
                n = perft(pos, d)
                print("%s\t%d\t%d" % (label, d, n))
                sys.stdout.flush()
                print(
                    "  %-22s depth %d: %12d nodes  %6.1fs" % (label, d, n, time.time() - t),
                    file=sys.stderr,
                )
    print("# suite finished in %.1fs" % (time.time() - total_start), file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fen", default=None)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--divide", action="store_true")
    ap.add_argument("--suite", default=None)
    args = ap.parse_args()

    sys.setrecursionlimit(10000)

    if args.suite:
        run_suite(args.suite)
        return

    pos = Position.startpos() if args.fen is None else Position.from_fen(args.fen)
    print(pos, file=sys.stderr)
    print("fen %s" % pos.to_fen(), file=sys.stderr)

    if args.divide:
        run_divide(pos, args.depth)
        return

    for d in range(1, args.depth + 1):
        t = time.time()
        n = perft(pos, d)
        print("%d\t%d" % (d, n))
        sys.stdout.flush()
        print("  depth %d: %d nodes in %.1fs" % (d, n, time.time() - t), file=sys.stderr)


if __name__ == "__main__":
    main()
