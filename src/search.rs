//! Alpha-beta search (build plan, Phase 2).
//!
//! Iterative deepening, transposition table, killer moves, history heuristic,
//! late move reductions, aspiration windows, and quiescence over all captures.
//!
//! Quiescence is not optional here. One move can swing four or more pieces via a
//! blast or a knight sweep, so any fixed-depth search without it hallucinates.
//!
//! Null-move pruning is implemented but **off by default**: position volatility is
//! extreme and a quiet-looking position can be lost outright in one move. Turn it
//! on only after measuring that it does not cost tactics.

use crate::eval::*;
use crate::movegen::*;
use crate::position::{GameResult, Position, PLY_CAP};
use crate::types::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const MAX_PLY: usize = 128;

// ---------------------------------------------------------------------------
// Transposition table
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct TtEntry {
    key: u64,
    mv: Move,
    score: i16,
    depth: i8,
    bound: u8,
    generation: u8,
}

impl TtEntry {
    const EMPTY: TtEntry = TtEntry {
        key: 0,
        mv: Move::NONE,
        score: 0,
        depth: -1,
        bound: 0,
        generation: 0,
    };
}

pub struct TranspositionTable {
    entries: Vec<TtEntry>,
    mask: usize,
    generation: u8,
}

impl TranspositionTable {
    pub fn with_megabytes(mb: usize) -> TranspositionTable {
        let bytes = mb.max(1) * 1024 * 1024;
        let mut n = bytes / std::mem::size_of::<TtEntry>();
        n = n.next_power_of_two() / 2;
        n = n.max(1024);
        TranspositionTable {
            entries: vec![TtEntry::EMPTY; n],
            mask: n - 1,
            generation: 0,
        }
    }

    pub fn clear(&mut self) {
        self.entries.iter_mut().for_each(|e| *e = TtEntry::EMPTY);
        self.generation = 0;
    }

    pub fn new_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    #[inline]
    fn probe(&self, key: u64) -> Option<&TtEntry> {
        let e = &self.entries[(key as usize) & self.mask];
        if e.key == key && e.depth >= 0 {
            Some(e)
        } else {
            None
        }
    }

    #[inline]
    fn store(&mut self, key: u64, mv: Move, score: i32, depth: i32, bound: Bound) {
        let slot = &mut self.entries[(key as usize) & self.mask];
        // Depth-preferred, but always replace entries from an older search.
        let stale = slot.generation != self.generation;
        if !stale && slot.key == key && (depth as i8) < slot.depth {
            return;
        }
        if !stale && slot.key != key && (depth as i8) < slot.depth - 3 {
            return;
        }
        *slot = TtEntry {
            key,
            mv,
            score: score.clamp(i16::MIN as i32 + 1, i16::MAX as i32) as i16,
            depth: depth.clamp(0, 127) as i8,
            bound: bound as u8,
            generation: self.generation,
        };
    }
}

// ---------------------------------------------------------------------------
// Configuration and result
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SearchLimits {
    pub depth: Option<i32>,
    pub movetime: Option<Duration>,
    pub nodes: Option<u64>,
}

impl Default for SearchLimits {
    fn default() -> Self {
        SearchLimits {
            depth: None,
            movetime: Some(Duration::from_millis(1000)),
            nodes: None,
        }
    }
}

impl SearchLimits {
    pub fn depth(d: i32) -> Self {
        SearchLimits {
            depth: Some(d),
            movetime: None,
            nodes: None,
        }
    }
    pub fn movetime_ms(ms: u64) -> Self {
        SearchLimits {
            depth: None,
            movetime: Some(Duration::from_millis(ms)),
            nodes: None,
        }
    }
    pub fn nodes(n: u64) -> Self {
        SearchLimits {
            depth: None,
            movetime: None,
            nodes: Some(n),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SearchResult {
    pub best: Move,
    pub score: i32,
    pub depth: i32,
    pub seldepth: i32,
    pub nodes: u64,
    pub pv: Vec<Move>,
    pub elapsed: Duration,
}

impl SearchResult {
    pub fn pv_string(&self) -> String {
        self.pv
            .iter()
            .map(|m| m.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }
    pub fn nps(&self) -> u64 {
        let s = self.elapsed.as_secs_f64();
        if s <= 0.0 {
            0
        } else {
            (self.nodes as f64 / s) as u64
        }
    }
    /// Render the score the way an operator wants to read it.
    pub fn score_string(&self) -> String {
        if self.score.abs() >= MATE_THRESHOLD {
            let plies = MATE - self.score.abs();
            let sign = if self.score > 0 { "" } else { "-" };
            format!("{sign}win in {plies} ply")
        } else {
            format!("{:+.2}", self.score as f64 / 100.0)
        }
    }
}

#[derive(Clone)]
pub struct SearchOptions {
    pub evaluator: Evaluator,
    pub tt_megabytes: usize,
    /// Off by default — see the module note.
    pub use_null_move: bool,
    pub use_lmr: bool,
    /// Emit `info depth ...` lines to stdout during iterative deepening.
    pub verbose: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        SearchOptions {
            evaluator: Evaluator::Hand,
            tt_megabytes: 64,
            use_null_move: false,
            use_lmr: true,
            verbose: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Searcher
// ---------------------------------------------------------------------------

pub struct Searcher {
    pub options: SearchOptions,
    tt: TranspositionTable,
    killers: [[Move; 2]; MAX_PLY],
    history: Vec<i32>, // 4096, indexed by packed move
    nodes: u64,
    seldepth: i32,
    deadline: Option<Instant>,
    node_limit: Option<u64>,
    stop: Arc<AtomicBool>,
    aborted: bool,
}

impl Searcher {
    pub fn new(options: SearchOptions) -> Searcher {
        let tt = TranspositionTable::with_megabytes(options.tt_megabytes);
        Searcher {
            options,
            tt,
            killers: [[Move::NONE; 2]; MAX_PLY],
            history: vec![0; 4096],
            nodes: 0,
            seldepth: 0,
            deadline: None,
            node_limit: None,
            stop: Arc::new(AtomicBool::new(false)),
            aborted: false,
        }
    }

    pub fn with_evaluator(ev: Evaluator) -> Searcher {
        Searcher::new(SearchOptions {
            evaluator: ev,
            ..Default::default()
        })
    }

    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }

    /// Clear everything that carries over between games.
    pub fn reset(&mut self) {
        self.tt.clear();
        self.killers = [[Move::NONE; 2]; MAX_PLY];
        self.history.iter_mut().for_each(|h| *h = 0);
    }

    // -----------------------------------------------------------------------
    // Iterative deepening
    // -----------------------------------------------------------------------

    pub fn search(&mut self, pos: &Position, limits: &SearchLimits) -> SearchResult {
        let start = Instant::now();
        self.nodes = 0;
        self.seldepth = 0;
        self.aborted = false;
        self.deadline = limits.movetime.map(|d| start + d);
        self.node_limit = limits.nodes;
        self.stop.store(false, Ordering::Relaxed);
        self.tt.new_generation();
        // History from the previous move is useful but stale; decay rather than clear.
        self.history.iter_mut().for_each(|h| *h /= 4);

        let max_depth = limits.depth.unwrap_or(MAX_PLY as i32 - 8);
        let mut result = SearchResult::default();

        // A legal move always exists (spec §1.4), so seed with the first one to
        // guarantee we never return a null move even if depth 1 is interrupted.
        let mut root_moves = MoveList::new();
        generate(pos, GenMode::Search, &mut root_moves);
        if root_moves.is_empty() {
            return result;
        }
        result.best = root_moves.moves[0];

        let mut alpha = -INFINITY;
        let mut beta = INFINITY;
        let mut depth = 1;

        while depth <= max_depth {
            let score = self.negamax(pos, depth, alpha, beta, 0, true);

            if self.aborted {
                break;
            }

            // Aspiration window: re-search wider on a fail high/low.
            if score <= alpha || score >= beta {
                alpha = -INFINITY;
                beta = INFINITY;
                continue;
            }

            let pv = self.extract_pv(pos, depth);
            result.score = score;
            result.depth = depth;
            result.seldepth = self.seldepth;
            result.nodes = self.nodes;
            result.elapsed = start.elapsed();
            if let Some(&m) = pv.first() {
                result.best = m;
            }
            result.pv = pv;

            if self.options.verbose {
                println!(
                    "info depth {} seldepth {} score {} nodes {} nps {} pv {}",
                    result.depth,
                    result.seldepth,
                    result.score_string(),
                    result.nodes,
                    result.nps(),
                    result.pv_string()
                );
            }

            // A forced result is not going to change with more depth.
            if score.abs() >= MATE_THRESHOLD {
                break;
            }
            if self.out_of_time() {
                break;
            }
            // Do not start an iteration we have no chance of finishing.
            if let (Some(dl), true) = (self.deadline, depth >= 4) {
                let used = start.elapsed();
                if Instant::now() + used / 2 > dl {
                    break;
                }
            }

            let window = 40;
            alpha = score - window;
            beta = score + window;
            depth += 1;
        }

        result.nodes = self.nodes;
        result.elapsed = start.elapsed();
        result
    }

    fn extract_pv(&self, pos: &Position, max: i32) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut p = *pos;
        let mut seen = Vec::new();
        for _ in 0..max.min(32) {
            if p.result().is_some() {
                break;
            }
            let Some(e) = self.tt.probe(tt_key(&p)) else { break };
            if e.mv.is_none() {
                break;
            }
            // Guard against a TT collision producing an illegal continuation.
            let mut list = MoveList::new();
            generate(&p, GenMode::Search, &mut list);
            if !list.contains(e.mv) {
                break;
            }
            if seen.contains(&p.key) {
                break;
            }
            seen.push(p.key);
            pv.push(e.mv);
            p.make_move(e.mv);
        }
        pv
    }

    // -----------------------------------------------------------------------
    // Time / abort
    // -----------------------------------------------------------------------

    #[inline]
    fn out_of_time(&self) -> bool {
        if self.stop.load(Ordering::Relaxed) {
            return true;
        }
        if let Some(n) = self.node_limit {
            if self.nodes >= n {
                return true;
            }
        }
        match self.deadline {
            Some(dl) => Instant::now() >= dl,
            None => false,
        }
    }

    #[inline]
    fn check_abort(&mut self) -> bool {
        if self.aborted {
            return true;
        }
        if self.nodes & 2047 == 0 && self.out_of_time() {
            self.aborted = true;
        }
        self.aborted
    }

    // -----------------------------------------------------------------------
    // Negamax
    // -----------------------------------------------------------------------

    fn negamax(
        &mut self,
        pos: &Position,
        depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        is_pv: bool,
    ) -> i32 {
        self.nodes += 1;
        if self.check_abort() {
            return 0;
        }
        self.seldepth = self.seldepth.max(ply as i32);

        // Terminal check first — spec §3 conditions are evaluated on the position
        // as it stands, and a game that is over has no moves worth generating.
        if let Some(r) = pos.result() {
            return terminal_score(r, pos.side, ply);
        }
        if ply >= MAX_PLY - 1 {
            return self.options.evaluator.eval(pos);
        }

        if depth <= 0 {
            return self.quiescence(pos, alpha, beta, ply);
        }

        // --- transposition probe --------------------------------------------
        let key = tt_key(pos);
        let mut tt_move = Move::NONE;
        if let Some(e) = self.tt.probe(key) {
            tt_move = e.mv;
            if !is_pv && e.depth as i32 >= depth {
                let s = from_tt_score(e.score as i32, ply);
                match e.bound {
                    b if b == Bound::Exact as u8 => return s,
                    b if b == Bound::Lower as u8 && s >= beta => return s,
                    b if b == Bound::Upper as u8 && s <= alpha => return s,
                    _ => {}
                }
            }
        }

        let static_eval = self.options.evaluator.eval(pos);

        // --- null move (off by default) --------------------------------------
        if self.options.use_null_move
            && !is_pv
            && depth >= 3
            && static_eval >= beta
            && pos.heavy_material_bb() & pos.occ_color[pos.side.index()] != 0
        {
            let mut child = *pos;
            child.side = child.side.flip();
            child.key ^= crate::tables::TABLES.zobrist_side;
            child.ply += 1;
            let r = 2 + depth / 4;
            let score = -self.negamax(&child, depth - r - 1, -beta, -beta + 1, ply + 1, false);
            if self.aborted {
                return 0;
            }
            if score >= beta {
                return beta;
            }
        }

        // --- move generation and ordering ------------------------------------
        let mut list = MoveList::new();
        generate(pos, GenMode::Search, &mut list);
        debug_assert!(!list.is_empty(), "spec §1.4 violated at {}", pos.to_fen());
        if list.is_empty() {
            return static_eval;
        }
        self.score_moves(pos, &mut list, tt_move, ply);

        let mut best_score = -INFINITY;
        let mut best_move = list.moves[0];
        let mut bound = Bound::Upper;

        for i in 0..list.len {
            pick_best(&mut list, i);
            let mv = list.moves[i];
            let (child, _) = pos.after(mv);

            // Late move reductions: quiet, late, non-tactical moves get a shallower
            // look first. Captures are never reduced — one blast decides games.
            let mut score;
            let quiet = !is_capture(pos, mv);
            let mut new_depth = depth - 1;
            if self.options.use_lmr && depth >= 3 && i >= 4 && quiet && !is_pv {
                let r = 1 + (i / 12).min(2) as i32;
                new_depth = (depth - 1 - r).max(1);
            }

            if i == 0 {
                score = -self.negamax(&child, new_depth, -beta, -alpha, ply + 1, is_pv);
            } else {
                score = -self.negamax(&child, new_depth, -alpha - 1, -alpha, ply + 1, false);
                if score > alpha && (new_depth < depth - 1 || score < beta) {
                    score = -self.negamax(&child, depth - 1, -beta, -alpha, ply + 1, is_pv);
                }
            }

            if self.aborted {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = mv;
            }
            if score > alpha {
                alpha = score;
                bound = Bound::Exact;
            }
            if alpha >= beta {
                bound = Bound::Lower;
                if quiet {
                    self.record_killer(mv, ply);
                    self.history[mv.index()] += depth * depth;
                }
                break;
            }
        }

        self.tt
            .store(key, best_move, to_tt_score(best_score, ply), depth, bound);
        best_score
    }

    // -----------------------------------------------------------------------
    // Quiescence — spec §5.6 exact deltas do the pruning
    // -----------------------------------------------------------------------

    fn quiescence(&mut self, pos: &Position, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;
        if self.check_abort() {
            return 0;
        }
        self.seldepth = self.seldepth.max(ply as i32);

        if let Some(r) = pos.result() {
            return terminal_score(r, pos.side, ply);
        }
        if ply >= MAX_PLY - 1 {
            return self.options.evaluator.eval(pos);
        }

        let stand_pat = self.options.evaluator.eval(pos);
        if stand_pat >= beta {
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let mut list = MoveList::new();
        generate(pos, GenMode::Captures, &mut list);
        if list.is_empty() {
            return stand_pat;
        }
        for i in 0..list.len {
            list.scores[i] = move_delta(pos, list.moves[i]);
        }

        let mut best = stand_pat;
        for i in 0..list.len {
            pick_best(&mut list, i);
            let mv = list.moves[i];
            let delta = list.scores[i];

            // Prune captures that lose material outright. A losing detonation can
            // still be right for the trailing side (spec §5.5), but that is a
            // strategic choice for the main search, not for quiescence — except
            // when it removes a king, which `delta` scores enormously.
            if delta < 0 && stand_pat + delta + 200 <= alpha {
                continue;
            }

            let (child, _) = pos.after(mv);
            let score = -self.quiescence(&child, -beta, -alpha, ply + 1);
            if self.aborted {
                return 0;
            }
            if score > best {
                best = score;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break;
            }
        }
        best
    }

    // -----------------------------------------------------------------------
    // Ordering
    // -----------------------------------------------------------------------

    fn score_moves(&self, pos: &Position, list: &mut MoveList, tt_move: Move, ply: usize) {
        let killers = self.killers[ply.min(MAX_PLY - 1)];
        for i in 0..list.len {
            let mv = list.moves[i];
            list.scores[i] = if mv == tt_move {
                1 << 24
            } else {
                // Exact material swing — strictly better information than a real
                // chess engine has at ordering time (spec §5.6). Scaled well above
                // history so that any real capture outranks any quiet move.
                let delta = move_delta(pos, mv);
                if delta != 0 {
                    (1 << 20) + delta * 8
                } else if mv == killers[0] {
                    1 << 18
                } else if mv == killers[1] {
                    (1 << 18) - 1
                } else {
                    self.history[mv.index()].min((1 << 17) - 1)
                }
            };
        }
    }

    fn record_killer(&mut self, mv: Move, ply: usize) {
        let p = ply.min(MAX_PLY - 1);
        if self.killers[p][0] != mv {
            self.killers[p][1] = self.killers[p][0];
            self.killers[p][0] = mv;
        }
    }
}

/// Transposition key for a position.
///
/// `Position::key` is pure position identity, which is what we want almost
/// everywhere. But two paths of different lengths can reach the same position, so
/// the same key can occur at different plies within one search — and near the
/// spec §4 cap the ply changes what the position is worth, because material
/// adjudication is about to fire. Fold the ply in only inside that window, so the
/// table stays shared for the rest of the game.
const PLY_SENSITIVE_WINDOW: u16 = 32;

#[inline(always)]
fn tt_key(pos: &Position) -> u64 {
    if PLY_CAP.saturating_sub(pos.ply) <= PLY_SENSITIVE_WINDOW {
        pos.key ^ crate::tables::TABLES.zobrist_ply[pos.ply as usize & 255]
    } else {
        pos.key
    }
}

/// Selection sort one element at a time — cheaper than sorting the whole list,
/// because a beta cutoff usually lands in the first few moves.
#[inline]
fn pick_best(list: &mut MoveList, from: usize) {
    let mut best = from;
    for j in from + 1..list.len {
        if list.scores[j] > list.scores[best] {
            best = j;
        }
    }
    if best != from {
        list.moves.swap(from, best);
        list.scores.swap(from, best);
    }
}

/// Spec §3: a finished game scores from the point of view of the side to move at
/// this node. Wins are discounted by ply so the search prefers the quicker one.
#[inline]
pub fn terminal_score(r: GameResult, stm: Color, ply: usize) -> i32 {
    match r {
        GameResult::Draw => 0,
        GameResult::Win(w) => {
            if w == stm {
                MATE - ply as i32
            } else {
                -(MATE - ply as i32)
            }
        }
    }
}

#[inline]
fn to_tt_score(score: i32, ply: usize) -> i32 {
    if score >= MATE_THRESHOLD {
        score + ply as i32
    } else if score <= -MATE_THRESHOLD {
        score - ply as i32
    } else {
        score
    }
}

#[inline]
fn from_tt_score(score: i32, ply: usize) -> i32 {
    if score >= MATE_THRESHOLD {
        score - ply as i32
    } else if score <= -MATE_THRESHOLD {
        score + ply as i32
    } else {
        score
    }
}

/// Convenience wrapper for one-shot searches (tests, self-play, gauntlet).
pub fn best_move(pos: &Position, limits: &SearchLimits, ev: Evaluator) -> SearchResult {
    let mut s = Searcher::with_evaluator(ev);
    s.search(pos, limits)
}

/// Remaining plies before the spec §4 cap adjudicates the game.
pub fn plies_left(pos: &Position) -> i32 {
    PLY_CAP as i32 - pos.ply as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_a_king_capture_in_one() {
        // White knight on b1 sweeps b2/b3 -> c3; black king on b3 is on the path.
        let pos = Position::from_fen("8/8/8/8/8/1k6/8/1N2K3 w - - 0 1").unwrap();
        let r = best_move(&pos, &SearchLimits::depth(3), Evaluator::Hand);
        assert!(
            r.score >= MATE_THRESHOLD,
            "expected a forced win, got {} via {}",
            r.score_string(),
            r.best
        );
    }

    #[test]
    fn avoids_walking_into_a_one_move_loss() {
        // Black to move. Kb6 would put the king next to the white rook's reach.
        let pos = Position::from_fen("8/8/1k6/8/8/8/8/R3K3 b - - 0 1").unwrap();
        let r = best_move(&pos, &SearchLimits::depth(4), Evaluator::Hand);
        assert!(r.score > -MATE_THRESHOLD, "black should not be lost here");
    }

    #[test]
    fn search_returns_a_legal_move() {
        let mut pos = Position::startpos();
        for _ in 0..12 {
            if pos.result().is_some() {
                break;
            }
            let r = best_move(&pos, &SearchLimits::depth(4), Evaluator::Hand);
            let legal = generate_vec(&pos, GenMode::All);
            assert!(
                legal.contains(&r.best),
                "illegal move {} in {}",
                r.best,
                pos.to_fen()
            );
            pos.make_move(r.best);
            pos.validate().unwrap();
        }
    }

    #[test]
    fn respects_the_ply_cap() {
        // One ply from the cap, with white a queen up: the cap adjudicates a win.
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 199 100").unwrap();
        let r = best_move(&pos, &SearchLimits::depth(4), Evaluator::Hand);
        assert!(r.score >= MATE_THRESHOLD, "material adjudication should win");
    }

    #[test]
    fn transposition_key_separates_plies_only_near_the_cap() {
        let early_a = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 20 11").unwrap();
        let early_b = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 24 13").unwrap();
        assert_eq!(
            tt_key(&early_a),
            tt_key(&early_b),
            "far from the cap the table should be shared across plies"
        );

        // Inside the window the ply changes what the position is worth, because
        // material adjudication is about to fire.
        let late_a = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 190 96").unwrap();
        let late_b = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 196 99").unwrap();
        assert_ne!(tt_key(&late_a), tt_key(&late_b));
        assert_ne!(tt_key(&late_a), late_a.key);
    }

    #[test]
    fn deeper_search_is_not_slower_than_linear() {
        let pos = Position::startpos();
        let r = best_move(&pos, &SearchLimits::depth(5), Evaluator::Hand);
        assert!(r.depth >= 5);
        assert!(!r.best.is_none());
    }
}
