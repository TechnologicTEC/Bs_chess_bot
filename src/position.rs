//! Board representation and move application.
//!
//! Copy-make: `Position` is `Copy`, so search stores a full 136-byte snapshot per
//! ply rather than an undo record. In a variant where one move can destroy nine
//! pieces, delta-undo is a bug farm for no measurable gain.

use crate::eval::PIECE_VALUE;
use crate::tables::TABLES;
use crate::types::*;

/// Spec §3.4 / §4 — hard cap at 200 plies.
pub const PLY_CAP: u16 = 200;

/// Spec §4 — material margin (centipawns) needed to claim the win at the cap.
pub const ADJUDICATION_THRESHOLD: i32 = 200;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameResult {
    Draw,
    Win(Color),
}

impl GameResult {
    /// +1 / 0 / -1 from `c`'s point of view.
    pub fn score_for(self, c: Color) -> f32 {
        match self {
            GameResult::Draw => 0.0,
            GameResult::Win(w) => {
                if w == c {
                    1.0
                } else {
                    -1.0
                }
            }
        }
    }
    pub fn describe(self) -> String {
        match self {
            GameResult::Draw => "draw".to_string(),
            GameResult::Win(c) => format!("{} wins", c.name()),
        }
    }
}

/// What a move destroyed. The `emptied` mask is the operator-mode diagnostic that
/// spec §8 identifies as the highest-value cross-check against the opposing engine.
#[derive(Clone, Copy, Debug, Default)]
pub struct MoveInfo {
    /// Every square that held a piece before the move and is empty after it.
    /// Includes the mover's origin square.
    pub emptied: Bitboard,
    /// Squares whose occupant was *destroyed* (excludes the mover's origin unless
    /// the mover itself died there).
    pub captured: Bitboard,
    /// True when a rook/bishop detonated and went up with the blast.
    pub mover_died: bool,
    pub piece: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub pieces: [Bitboard; NUM_PIECES],
    pub occ_color: [Bitboard; 2],
    pub occ: Bitboard,
    pub side: Color,
    /// Plies played since the start of the game (spec §4 ply cap counts these).
    pub ply: u16,
    pub key: u64,
}

impl Position {
    pub fn empty() -> Position {
        Position {
            pieces: [0; NUM_PIECES],
            occ_color: [0; 2],
            occ: 0,
            side: Color::White,
            ply: 0,
            key: 0,
        }
    }

    pub fn startpos() -> Position {
        Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1").unwrap()
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    #[inline(always)]
    pub fn piece_at(&self, s: usize) -> Option<usize> {
        if self.occ & bb(s) == 0 {
            return None;
        }
        let b = bb(s);
        (0..NUM_PIECES).find(|&p| self.pieces[p] & b != 0)
    }

    #[inline(always)]
    pub fn bb_of(&self, c: Color, pt: PieceType) -> Bitboard {
        self.pieces[piece_index(c, pt)]
    }

    #[inline(always)]
    pub fn king_sq(&self, c: Color) -> Option<usize> {
        let b = self.bb_of(c, PieceType::King);
        if b == 0 {
            None
        } else {
            Some(b.trailing_zeros() as usize)
        }
    }

    #[inline(always)]
    pub fn empty_squares(&self) -> Bitboard {
        !self.occ
    }

    /// Material in centipawns for one side, kings excluded.
    pub fn material_cp(&self, c: Color) -> i32 {
        let mut total = 0;
        for pt in [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ] {
            total += self.bb_of(c, pt).count_ones() as i32 * PIECE_VALUE[pt.index()];
        }
        total
    }

    /// Everything that is not a king and not a pawn — the material whose
    /// disappearance triggers the spec §3.3 draw.
    #[inline(always)]
    pub fn heavy_material_bb(&self) -> Bitboard {
        let mut b = 0;
        for c in [Color::White, Color::Black] {
            for pt in [
                PieceType::Knight,
                PieceType::Bishop,
                PieceType::Rook,
                PieceType::Queen,
            ] {
                b |= self.bb_of(c, pt);
            }
        }
        b
    }

    // -----------------------------------------------------------------------
    // Terminal conditions — spec §3, evaluated in order
    // -----------------------------------------------------------------------

    pub fn result(&self) -> Option<GameResult> {
        let wk = self.bb_of(Color::White, PieceType::King) != 0;
        let bk = self.bb_of(Color::Black, PieceType::King) != 0;

        // 1. Both kings removed simultaneously -> draw.
        if !wk && !bk {
            return Some(GameResult::Draw);
        }
        // 2. Exactly one king removed -> the surviving king's side wins. This
        //    includes blowing up your own king.
        if !wk {
            return Some(GameResult::Win(Color::Black));
        }
        if !bk {
            return Some(GameResult::Win(Color::White));
        }
        // 3. Only kings and pawns remain -> draw.
        if self.heavy_material_bb() == 0 {
            return Some(GameResult::Draw);
        }
        // 4. Ply cap -> adjudicate by material (spec §4).
        if self.ply >= PLY_CAP {
            return Some(self.adjudicate());
        }
        None
    }

    /// Spec §4: score the position by material rather than calling an automatic
    /// draw, so that a losing side gains nothing by shuffling to the cap.
    pub fn adjudicate(&self) -> GameResult {
        let diff = self.material_cp(Color::White) - self.material_cp(Color::Black);
        if diff > ADJUDICATION_THRESHOLD {
            GameResult::Win(Color::White)
        } else if diff < -ADJUDICATION_THRESHOLD {
            GameResult::Win(Color::Black)
        } else {
            GameResult::Draw
        }
    }

    // -----------------------------------------------------------------------
    // Mutation primitives
    // -----------------------------------------------------------------------

    pub fn put_piece(&mut self, p: usize, s: usize) {
        debug_assert!(self.occ & bb(s) == 0);
        self.pieces[p] |= bb(s);
        self.occ_color[piece_color(p).index()] |= bb(s);
        self.occ |= bb(s);
        self.key ^= TABLES.zobrist_piece[p][s];
    }

    #[inline(always)]
    fn remove_known(&mut self, p: usize, s: usize) {
        self.pieces[p] &= !bb(s);
        self.occ_color[piece_color(p).index()] &= !bb(s);
        self.occ &= !bb(s);
        self.key ^= TABLES.zobrist_piece[p][s];
    }

    #[inline(always)]
    fn move_known(&mut self, p: usize, from: usize, to: usize) {
        let delta = bb(from) | bb(to);
        self.pieces[p] ^= delta;
        self.occ_color[piece_color(p).index()] ^= delta;
        self.occ ^= delta;
        self.key ^= TABLES.zobrist_piece[p][from] ^ TABLES.zobrist_piece[p][to];
    }

    /// Destroy every piece inside `mask`, of either colour and any type.
    /// Idempotent on empty squares.
    #[inline]
    fn clear_mask(&mut self, mask: Bitboard) {
        let hit = self.occ & mask;
        if hit == 0 {
            return;
        }
        for p in 0..NUM_PIECES {
            let x = self.pieces[p] & hit;
            if x != 0 {
                for s in bits(x) {
                    self.key ^= TABLES.zobrist_piece[p][s];
                }
                self.pieces[p] &= !x;
            }
        }
        self.occ_color[0] &= !hit;
        self.occ_color[1] &= !hit;
        self.occ &= !hit;
    }

    // -----------------------------------------------------------------------
    // Move application — spec §2
    // -----------------------------------------------------------------------

    /// Applies `mv` in place. `mv` must be pseudo-legal for the side to move;
    /// every pseudo-legal move is legal (spec §1.1), so there is no filtering.
    pub fn make_move(&mut self, mv: Move) -> MoveInfo {
        let (from, to) = (mv.from(), mv.to());
        let p = match self.piece_at(from) {
            Some(p) => p,
            None => panic!("make_move: no piece on {}", square_name(from)),
        };
        debug_assert_eq!(piece_color(p), self.side, "moving the wrong colour");
        let occ_before = self.occ;
        let mut mover_died = false;
        // Computed per branch rather than as `occ_before & !occ_after`: a knight,
        // queen or king that lands on the square it just cleared leaves that
        // square occupied afterwards, so set-difference would silently lose the
        // capture. That is exactly the class of bug spec §8 warns about.
        let captured;

        match piece_type_of(p) {
            // §2.1 — teleport to any empty square. Never captures.
            PieceType::Pawn => {
                debug_assert!(self.occ & bb(to) == 0, "pawn teleport onto occupied square");
                captured = 0;
                self.move_known(p, from, to);
            }

            // §2.2 — sweep: destroy both intermediates and the destination,
            // then land. Unblockable; the knight survives.
            PieceType::Knight => {
                let path = TABLES.knight_path_lookup[from][to];
                debug_assert!(path != 0, "illegal knight move {mv}");
                captured = path & occ_before;
                self.clear_mask(path);
                self.move_known(p, from, to);
            }

            // §2.3 — one-shot detonator. Landing on an occupied square clears the
            // 3x3 centred on it, and takes the mover with it. No chain reactions.
            PieceType::Bishop | PieceType::Rook => {
                if self.occ & bb(to) != 0 {
                    captured = TABLES.blast[to] & occ_before & !bb(from);
                    self.remove_known(p, from);
                    self.clear_mask(TABLES.blast[to]);
                    mover_died = true;
                } else {
                    captured = 0;
                    self.move_known(p, from, to);
                }
            }

            // §2.4 — forced sweeper. Everything passed over is captured, plus the
            // landing square if occupied. The queen survives.
            PieceType::Queen => {
                debug_assert!(
                    TABLES.dir_between[from][to] != 8,
                    "queen move {mv} is not along a ray"
                );
                let swept = (TABLES.between[from][to] | bb(to)) & occ_before;
                captured = swept;
                self.clear_mask(swept);
                self.move_known(p, from, to);
            }

            // §2.5 — displacement capture only.
            PieceType::King => {
                captured = bb(to) & occ_before;
                self.clear_mask(bb(to));
                self.move_known(p, from, to);
            }
        }

        self.side = self.side.flip();
        self.key ^= TABLES.zobrist_side;
        self.ply += 1;

        MoveInfo {
            emptied: occ_before & !self.occ,
            captured,
            mover_died,
            piece: p,
        }
    }

    /// Non-mutating variant: returns the child position alongside the info record.
    pub fn after(&self, mv: Move) -> (Position, MoveInfo) {
        let mut child = *self;
        let info = child.make_move(mv);
        (child, info)
    }

    /// Recompute the Zobrist key from scratch (used by tests and `setfen`).
    pub fn recompute_key(&mut self) {
        let mut k = 0u64;
        for p in 0..NUM_PIECES {
            for s in bits(self.pieces[p]) {
                k ^= TABLES.zobrist_piece[p][s];
            }
        }
        if self.side == Color::Black {
            k ^= TABLES.zobrist_side;
        }
        self.key = k;
    }

    /// Internal consistency check — occupancy agrees with the piece boards and no
    /// square holds two pieces.
    pub fn validate(&self) -> Result<(), String> {
        let mut acc = 0u64;
        for p in 0..NUM_PIECES {
            if acc & self.pieces[p] != 0 {
                return Err(format!("two pieces on {}", square_list(acc & self.pieces[p])));
            }
            acc |= self.pieces[p];
        }
        if acc != self.occ {
            return Err("occupancy disagrees with piece boards".into());
        }
        let mut w = 0u64;
        let mut b = 0u64;
        for p in 0..NUM_PIECES {
            if piece_color(p) == Color::White {
                w |= self.pieces[p];
            } else {
                b |= self.pieces[p];
            }
        }
        if w != self.occ_color[0] || b != self.occ_color[1] {
            return Err("colour occupancy disagrees with piece boards".into());
        }
        if self.bb_of(Color::White, PieceType::King).count_ones() > 1
            || self.bb_of(Color::Black, PieceType::King).count_ones() > 1
        {
            return Err("more than one king of a colour".into());
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // FEN — spec §8. Castling and en-passant fields are always `-`; the ply
    // count lives in the halfmove-clock field.
    // -----------------------------------------------------------------------

    pub fn from_fen(fen: &str) -> Result<Position, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.is_empty() {
            return Err("empty FEN".into());
        }
        let mut pos = Position::empty();

        let mut rank = 7usize;
        let mut file = 0usize;
        for ch in parts[0].chars() {
            match ch {
                '/' => {
                    if file != 8 {
                        return Err(format!("rank {} has {file} files, expected 8", rank + 1));
                    }
                    if rank == 0 {
                        return Err("too many ranks in FEN".into());
                    }
                    rank -= 1;
                    file = 0;
                }
                '1'..='8' => file += ch as usize - '0' as usize,
                _ => {
                    let pt = PieceType::from_char(ch)
                        .ok_or_else(|| format!("bad piece char '{ch}' in FEN"))?;
                    let color = if ch.is_ascii_uppercase() {
                        Color::White
                    } else {
                        Color::Black
                    };
                    if file > 7 {
                        return Err("rank overflow in FEN".into());
                    }
                    pos.put_piece(piece_index(color, pt), sq(rank, file));
                    file += 1;
                }
            }
        }
        if rank != 0 || file != 8 {
            return Err("FEN board section is not 8x8".into());
        }

        pos.side = match parts.get(1).copied().unwrap_or("w") {
            "w" => Color::White,
            "b" => Color::Black,
            other => return Err(format!("bad side-to-move '{other}'")),
        };
        // parts[2] castling, parts[3] en passant — always '-', ignored.
        pos.ply = parts
            .get(4)
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(0);

        pos.recompute_key();
        pos.validate()?;
        Ok(pos)
    }

    pub fn to_fen(&self) -> String {
        let mut out = String::new();
        for rank in (0..8).rev() {
            let mut run = 0;
            for file in 0..8 {
                match self.piece_at(sq(rank, file)) {
                    None => run += 1,
                    Some(p) => {
                        if run > 0 {
                            out.push_str(&run.to_string());
                            run = 0;
                        }
                        let c = piece_type_of(p).to_char();
                        out.push(if piece_color(p) == Color::White {
                            c.to_ascii_uppercase()
                        } else {
                            c
                        });
                    }
                }
            }
            if run > 0 {
                out.push_str(&run.to_string());
            }
            if rank > 0 {
                out.push('/');
            }
        }
        let stm = if self.side == Color::White { 'w' } else { 'b' };
        let fullmove = self.ply / 2 + 1;
        format!("{out} {stm} - - {} {}", self.ply, fullmove)
    }

    // -----------------------------------------------------------------------
    // Display
    // -----------------------------------------------------------------------

    pub fn board_string(&self) -> String {
        let mut out = String::new();
        out.push_str("  +-----------------+\n");
        for rank in (0..8).rev() {
            out.push_str(&format!("{} | ", rank + 1));
            for file in 0..8 {
                let ch = match self.piece_at(sq(rank, file)) {
                    None => '.',
                    Some(p) => {
                        let c = piece_type_of(p).to_char();
                        if piece_color(p) == Color::White {
                            c.to_ascii_uppercase()
                        } else {
                            c
                        }
                    }
                };
                out.push(ch);
                out.push(' ');
            }
            out.push_str("|\n");
        }
        out.push_str("  +-----------------+\n");
        out.push_str("    a b c d e f g h\n");
        out
    }
}

impl std::fmt::Debug for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\n{}", self.board_string(), self.to_fen())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_roundtrips() {
        let p = Position::startpos();
        assert_eq!(
            p.to_fen(),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
        );
        assert_eq!(p.occ.count_ones(), 32);
        assert!(p.validate().is_ok());
        assert_eq!(p.result(), None);
    }

    #[test]
    fn fen_roundtrip_arbitrary() {
        for fen in [
            "8/8/4k3/8/8/3K4/8/8 b - - 41 21",
            "r3k3/8/8/8/8/8/8/4K2R w - - 0 1",
        ] {
            let p = Position::from_fen(fen).unwrap();
            assert_eq!(p.to_fen(), fen);
        }
    }

    #[test]
    fn key_matches_after_moves() {
        let mut p = Position::startpos();
        p.make_move(Move::parse("b1c3").unwrap());
        p.make_move(Move::parse("g8f6").unwrap());
        let mut q = p;
        q.recompute_key();
        assert_eq!(p.key, q.key);
    }
}
