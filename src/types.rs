//! Core scalar types: squares, colours, pieces, moves.
//!
//! Square indexing is `sq = rank * 8 + file`, with rank 0 = rank "1" and file 0 = file "a".
//! So a1 = 0, h1 = 7, a8 = 56, h8 = 63. Spec coordinates `(r, c)` map to `(rank, file)`.

pub type Bitboard = u64;

pub const NUM_SQUARES: usize = 64;

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline(always)]
    pub fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
    #[inline(always)]
    pub fn index(self) -> usize {
        self as usize
    }
    pub fn from_index(i: usize) -> Color {
        if i == 0 {
            Color::White
        } else {
            Color::Black
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Color::White => "white",
            Color::Black => "black",
        }
    }
}

// ---------------------------------------------------------------------------
// Piece type
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

pub const NUM_PIECE_TYPES: usize = 6;
pub const ALL_PIECE_TYPES: [PieceType; 6] = [
    PieceType::Pawn,
    PieceType::Knight,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Queen,
    PieceType::King,
];

impl PieceType {
    #[inline(always)]
    pub fn index(self) -> usize {
        self as usize
    }
    pub fn from_index(i: usize) -> PieceType {
        ALL_PIECE_TYPES[i]
    }
    pub fn from_char(c: char) -> Option<PieceType> {
        Some(match c.to_ascii_lowercase() {
            'p' => PieceType::Pawn,
            'n' => PieceType::Knight,
            'b' => PieceType::Bishop,
            'r' => PieceType::Rook,
            'q' => PieceType::Queen,
            'k' => PieceType::King,
            _ => return None,
        })
    }
    pub fn to_char(self) -> char {
        match self {
            PieceType::Pawn => 'p',
            PieceType::Knight => 'n',
            PieceType::Bishop => 'b',
            PieceType::Rook => 'r',
            PieceType::Queen => 'q',
            PieceType::King => 'k',
        }
    }
}

/// Combined piece index: `color * 6 + piece_type`, range 0..12.
pub const NUM_PIECES: usize = 12;

#[inline(always)]
pub fn piece_index(c: Color, pt: PieceType) -> usize {
    c.index() * NUM_PIECE_TYPES + pt.index()
}

#[inline(always)]
pub fn piece_color(pi: usize) -> Color {
    Color::from_index(pi / NUM_PIECE_TYPES)
}

#[inline(always)]
pub fn piece_type_of(pi: usize) -> PieceType {
    PieceType::from_index(pi % NUM_PIECE_TYPES)
}

// ---------------------------------------------------------------------------
// Squares
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn sq(rank: usize, file: usize) -> usize {
    rank * 8 + file
}

#[inline(always)]
pub fn rank_of(s: usize) -> usize {
    s >> 3
}

#[inline(always)]
pub fn file_of(s: usize) -> usize {
    s & 7
}

#[inline(always)]
pub fn bb(s: usize) -> Bitboard {
    1u64 << s
}

/// `e2` -> 12. Returns `None` for anything that is not two chars in range.
pub fn parse_square(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    if b.len() != 2 {
        return None;
    }
    let file = (b[0] as char).to_ascii_lowercase() as i32 - 'a' as i32;
    let rank = (b[1] as char) as i32 - '1' as i32;
    if !(0..8).contains(&file) || !(0..8).contains(&rank) {
        return None;
    }
    Some(sq(rank as usize, file as usize))
}

pub fn square_name(s: usize) -> String {
    let f = (b'a' + file_of(s) as u8) as char;
    let r = (b'1' + rank_of(s) as u8) as char;
    format!("{f}{r}")
}

// ---------------------------------------------------------------------------
// Moves
// ---------------------------------------------------------------------------

/// A move is fully described by `(from, to)` — spec §1.2. No promotion, castling
/// or en-passant flags exist, so 12 bits is a complete encoding.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Move(pub u16);

impl Move {
    #[inline(always)]
    pub fn new(from: usize, to: usize) -> Move {
        Move((from as u16) | ((to as u16) << 6))
    }
    #[inline(always)]
    pub fn from(self) -> usize {
        (self.0 & 63) as usize
    }
    #[inline(always)]
    pub fn to(self) -> usize {
        ((self.0 >> 6) & 63) as usize
    }
    /// Sentinel for "no move". `a1a1` is never legal (a pawn may not teleport to
    /// its own square, and no other piece can move zero distance).
    pub const NONE: Move = Move(0);
    #[inline(always)]
    pub fn is_none(self) -> bool {
        self.0 == 0
    }
    /// Index into a 4096-entry move-indexed table.
    #[inline(always)]
    pub fn index(self) -> usize {
        (self.0 & 0x0fff) as usize
    }

    pub fn parse(s: &str) -> Option<Move> {
        if s.len() != 4 {
            return None;
        }
        let from = parse_square(&s[0..2])?;
        let to = parse_square(&s[2..4])?;
        Some(Move::new(from, to))
    }
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", square_name(self.from()), square_name(self.to()))
    }
}

impl std::fmt::Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

// ---------------------------------------------------------------------------
// Bitboard helpers
// ---------------------------------------------------------------------------

/// Iterate set bits of a bitboard, lowest first.
pub struct BitIter(pub Bitboard);

impl Iterator for BitIter {
    type Item = usize;
    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            let s = self.0.trailing_zeros() as usize;
            self.0 &= self.0 - 1;
            Some(s)
        }
    }
}

#[inline(always)]
pub fn bits(b: Bitboard) -> BitIter {
    BitIter(b)
}

pub fn bitboard_string(b: Bitboard) -> String {
    let mut out = String::new();
    for r in (0..8).rev() {
        for f in 0..8 {
            out.push(if b & bb(sq(r, f)) != 0 { 'X' } else { '.' });
        }
        out.push('\n');
    }
    out
}

/// Human-readable, file-then-rank sorted list of squares in a mask.
pub fn square_list(b: Bitboard) -> String {
    let mut v: Vec<String> = bits(b).map(square_name).collect();
    v.sort();
    v.join(" ")
}
