#!/usr/bin/env python3
"""
Marvin UCI Chess Engine (ONNX Runtime Version)

A self-contained UCI engine that uses ONNX Runtime for inference instead of PyTorch.
This enables distribution as a standalone executable without PyTorch dependencies.

Features:
- Full UCI protocol support
- ONNX Runtime inference with GPU (CUDA) or CPU
- Temperature-based move sampling with top-p (nucleus) filtering
- Time prediction for simulated thinking
- MCTS search with tree reuse
- All UCI options from the PyTorch version

Usage:
    python -m inference.uci_onnx

Requirements:
    pip install onnxruntime-gpu  # For GPU support
    pip install onnxruntime      # For CPU-only
    pip install numpy chess
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import chess


# ==============================================================================
# Constants and Configuration
# ==============================================================================

HISTORY_LEN = 8
NUM_SQUARES = 64
NUM_POLICY_OUTPUTS = 4098

# Piece encoding matching process_pgn.py
PIECE_MAP = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

# Time control categories
TC_BLITZ = 0
TC_RAPID = 1
TC_CLASSICAL = 2

# Known increment values the model was trained on
KNOWN_INCREMENTS = [0, 2, 3, 5, 10]

# Promotion indices
PROMO_INDEX = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}
PROMO_INDEX_TO_CHAR = {0: "q", 1: "r", 2: "b", 3: "n"}
PROMO_CHAR_TO_PIECE = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}

# Default settings
DEFAULT_GAME_SETTINGS: dict = {
    "temperature": 0.9,
    "time_temperature": 0.5,
    "top_p": 0.95,
    "time_top_p": 0.95,
    "opening_temperature": 1.2,
    "opening_length": 10,
    "human_elo": 2400,
    "engine_elo": 2400,
    "human_color": chess.WHITE,
    "simulate_thinking_time": False,
    "internal_clock": False,
    "use_real_time": False,
    "use_mode_time": False,
    "use_expected_time": True,
    "start_clock_s": 300.0,
    "inc_s": 0.0,
    # MCTS
    "use_mcts": False,
    "mcts_simulations": 256,
    "mcts_c_puct": 2.0,
    "mcts_max_children": 48,
    "mcts_root_dirichlet_alpha": 0.0,
    "mcts_root_exploration_frac": 0.0,
    "mcts_final_temperature": 0.0,
    "mcts_final_top_p": 0.90,
    "mcts_max_depth": 96,
    "mcts_leaf_batch_size": 1,
    "mcts_adaptive": True,
    "mcts_adaptive_scale": 150.0,
    "mcts_contempt": 0.15,
    "mcts_fpu_reduction": 0.20,
    "mcts_simulate_time": False,
    "mcts_start_ply": 0,
    "mcts_tree_reuse": True,
}

START_CLOCK_S = 300.0
DEFAULT_RNG_SEED = 67


# ==============================================================================
# Utility Functions
# ==============================================================================

def _bool_from_uci(value: str) -> bool:
    v = str(value).strip().lower()
    return v in ("1", "true", "yes", "on")


def _mirror_square(sq: int) -> int:
    return sq ^ 56


def _canonical_to_real_move(move: chess.Move, real_turn: chess.Color) -> chess.Move:
    if real_turn == chess.WHITE:
        return move
    return chess.Move(
        _mirror_square(move.from_square),
        _mirror_square(move.to_square),
        promotion=move.promotion,
    )


def clamp_to_known_increment(inc_s: float) -> float:
    """Clamp an increment to the nearest known value for better model generalization."""
    return float(min(KNOWN_INCREMENTS, key=lambda x: abs(x - inc_s)))


def get_tc_category(base_seconds: float, inc_seconds: float) -> int:
    """Determine time control category: Blitz, Rapid, or Classical."""
    duration = float(base_seconds) + 40.0 * float(inc_seconds)
    if duration < 600.0:
        return TC_BLITZ
    if duration < 1800.0:
        return TC_RAPID
    return TC_CLASSICAL


def canonicalize(board: chess.Board) -> chess.Board:
    """Return board from white's perspective (mirror if black to move)."""
    return board if board.turn == chess.WHITE else board.mirror()


def encode_board(board: chess.Board) -> List[int]:
    """Encode board state as 64 integers."""
    tokens = [0] * NUM_SQUARES
    for square in range(NUM_SQUARES):
        piece = board.piece_at(square)
        if piece is None:
            continue
        val = PIECE_MAP[piece.piece_type]
        tokens[square] = val if piece.color == chess.WHITE else val + 6
    return tokens


def encode_legal_moves(board: chess.Board) -> List[Tuple[int, int, int]]:
    """Encode legal moves as (from_sq, to_sq, promo) tuples."""
    promo_map = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}
    out: List[Tuple[int, int, int]] = []
    for mv in board.legal_moves:
        out.append((mv.from_square, mv.to_square, promo_map.get(mv.promotion, 0)))
    return out


def _time_bin_to_seconds(bin_idx: int, active_clock_s: float) -> float:
    """Convert time bin index to seconds."""
    scaled_mid = (bin_idx + 0.5) / 256.0
    return float((scaled_mid ** 2) * max(1e-6, active_clock_s))


# ==============================================================================
# Board History and Encoding
# ==============================================================================

def build_history_from_position(
    start_board: chess.Board,
    moves_uci: List[str],
    *,
    history_len: int = HISTORY_LEN,
) -> Tuple[chess.Board, List[List[int]], List[int]]:
    """Replay moves and return (final_board, board_history, repetition_flags)."""
    from collections import deque
    
    board = start_board.copy(stack=False)
    hist: deque = deque(maxlen=history_len)
    raw: deque = deque(maxlen=history_len)
    
    canonical = canonicalize(board)
    hist.appendleft(encode_board(canonical))
    raw.appendleft(canonical.copy(stack=False))
    
    for uci in moves_uci:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            raise ValueError(f"Illegal move in position list: {uci}")
        board.push(mv)
        canonical = canonicalize(board)
        hist.appendleft(encode_board(canonical))
        raw.appendleft(canonical.copy(stack=False))
    
    board_history = list(hist)
    while len(board_history) < history_len:
        board_history.append([0] * NUM_SQUARES)
    board_history = board_history[:history_len]
    
    # Repetition flags
    seen: dict = {}
    flags: List[int] = []
    for b in list(raw):
        key = b.board_fen()
        seen[key] = seen.get(key, 0) + 1
        flags.append(1 if seen[key] >= 2 else 0)
    while len(flags) < history_len:
        flags.insert(0, 0)
    
    return board, board_history, flags[:history_len]


def _history_from_board(board: chess.Board, *, history_len: int = HISTORY_LEN) -> Tuple[List[List[int]], List[int]]:
    """Build board history from a board with move stack."""
    b = board.copy(stack=True)
    boards: List[chess.Board] = []
    
    for _ in range(history_len):
        boards.append(canonicalize(b).copy(stack=False))
        if not b.move_stack:
            break
        b.pop()
    
    board_history = [encode_board(x) for x in boards]
    while len(board_history) < history_len:
        board_history.append([0] * 64)
    board_history = board_history[:history_len]
    
    seen: dict = {}
    flags: List[int] = []
    for bb in boards:
        key = bb.board_fen()
        seen[key] = seen.get(key, 0) + 1
        flags.append(1 if seen[key] >= 2 else 0)
    while len(flags) < history_len:
        flags.insert(0, 0)
    
    return board_history, flags[:history_len]


# ==============================================================================
# Context and Batch Creation (NumPy-based, no PyTorch)
# ==============================================================================

@dataclass
class ContextOptions:
    active_elo: int = 1900
    opponent_elo: int = 1900
    active_clock_s: float = 300.0
    opponent_clock_s: float = 300.0
    active_inc_s: float = 0.0
    opponent_inc_s: float = 0.0
    tc_base_s: Optional[float] = None
    halfmove_clock: int = 0


def make_model_batch_numpy(
    *,
    board: chess.Board,
    board_history: List[List[int]],
    repetition_flags: List[int],
    time_history_s: Optional[List[float]] = None,
    ctx: ContextOptions,
    tc_cat: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Build the tensor batch as numpy arrays (for ONNX Runtime)."""
    
    if time_history_s is None:
        time_history_s = [0.0] * HISTORY_LEN
    if len(time_history_s) != HISTORY_LEN:
        raise ValueError(f"time_history_s must have len {HISTORY_LEN}")
    
    # Canonical board for legal-move encoding
    canonical = canonicalize(board)
    legal_moves = encode_legal_moves(canonical)
    
    legal_mask = np.zeros(NUM_POLICY_OUTPUTS, dtype=np.bool_)
    for from_sq, to_sq, _promo in legal_moves:
        legal_mask[from_sq * 64 + to_sq] = True
    
    # Scalars (must match dataset.py normalization)
    active_elo_norm = (ctx.active_elo - 1900) / 700.0
    opp_elo_norm = (ctx.opponent_elo - 1900) / 700.0
    ply_norm = board.fullmove_number * 2 - (0 if board.turn == chess.WHITE else 1)
    ply_norm = ply_norm / 100.0
    
    active_clock_norm = math.log1p(max(0.0, ctx.active_clock_s)) / 10.0
    opp_clock_norm = math.log1p(max(0.0, ctx.opponent_clock_s)) / 10.0
    
    clamped_active_inc = clamp_to_known_increment(ctx.active_inc_s)
    clamped_opp_inc = clamp_to_known_increment(ctx.opponent_inc_s)
    active_inc_norm = clamped_active_inc / 30.0
    opp_inc_norm = clamped_opp_inc / 30.0
    hmc_norm = float(ctx.halfmove_clock) / 100.0
    
    scalars = np.array([
        active_elo_norm, opp_elo_norm, ply_norm,
        active_clock_norm, opp_clock_norm,
        active_inc_norm, opp_inc_norm, hmc_norm
    ], dtype=np.float32)
    
    if tc_cat is None:
        base_s = float(ctx.tc_base_s) if ctx.tc_base_s is not None else float(max(ctx.active_clock_s, ctx.opponent_clock_s))
        tc_cat = get_tc_category(base_s, ctx.active_inc_s)
    
    castling = np.array([
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
    ], dtype=np.float32)
    
    ep_mask = np.zeros(64, dtype=np.float32)
    if board.ep_square is not None:
        ep_mask[int(board.ep_square)] = 1.0
    
    batch = {
        "board_history": np.array(board_history, dtype=np.int64)[np.newaxis, ...],
        "time_history": (np.array(time_history_s, dtype=np.float32) / 60.0)[np.newaxis, ...],
        "rep_flags": np.array(repetition_flags, dtype=np.float32)[np.newaxis, ...],
        "castling": castling[np.newaxis, ...],
        "ep_mask": ep_mask[np.newaxis, ...],
        "scalars": scalars[np.newaxis, ...],
        "tc_cat": np.array([int(tc_cat)], dtype=np.int64),
        "legal_mask": legal_mask[np.newaxis, ...],
    }
    
    return batch


# ==============================================================================
# Sampling Functions (NumPy-based)
# ==============================================================================

@dataclass(frozen=True)
class SampleResult:
    move_index: int
    prob: float


def _top_p_filter(probs: np.ndarray, top_p: float) -> np.ndarray:
    if top_p >= 1.0:
        return probs
    if top_p <= 0.0:
        out = np.zeros_like(probs)
        out[int(np.argmax(probs))] = 1.0
        return out
    
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    cdf = np.cumsum(sorted_probs)
    
    keep = cdf <= top_p
    if not np.any(keep):
        keep[0] = True
    else:
        first_over = int(np.argmax(cdf > top_p))
        keep[first_over] = True
    
    mask = np.zeros_like(probs, dtype=bool)
    mask[order[keep]] = True
    out = np.where(mask, probs, 0.0)
    s = float(out.sum())
    return out / s if s > 0 else probs


def sample_from_logits_numpy(
    logits: np.ndarray,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> SampleResult:
    """Sample a single index from logits using temperature + nucleus (top-p)."""
    if rng is None:
        rng = np.random.default_rng()
    
    x = logits.astype(np.float64)
    if temperature <= 0.0:
        idx = int(np.argmax(x))
        return SampleResult(move_index=idx, prob=1.0)
    
    x = x / float(temperature)
    x = x - x.max()
    probs = np.exp(x)
    probs = probs / (probs.sum() + 1e-12)
    
    probs = _top_p_filter(probs, float(top_p))
    idx = int(rng.choice(len(probs), p=probs))
    return SampleResult(move_index=idx, prob=float(probs[idx]))


def select_promo_numpy(
    promo_logits_row: np.ndarray,
    *,
    temperature: float,
    top_p: float,
    rng: Optional[np.random.Generator] = None,
) -> str:
    """Select promotion piece char from a 4-logit vector (q,r,b,n)."""
    res = sample_from_logits_numpy(promo_logits_row, temperature=temperature, top_p=top_p, rng=rng)
    return PROMO_INDEX_TO_CHAR.get(res.move_index, "q")


# ==============================================================================
# Policy Output
# ==============================================================================

@dataclass
class PolicyOutput:
    move: Optional[chess.Move]
    policy_prob: float
    is_resign: bool = False
    is_flag: bool = False
    stats: Optional[dict] = None


# ==============================================================================
# ONNX Runtime Session Management
# ==============================================================================

class OnnxInferenceSession:
    """Wrapper for ONNX Runtime inference session with GPU support."""
    
    def __init__(self, model_path: Path, prefer_gpu: bool = True):
        import onnxruntime as ort
        
        self.model_path = model_path
        
        # Select execution providers
        providers = []
        if prefer_gpu:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        # Session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(str(model_path), sess_options, providers=providers)
        self.active_provider = self.session.get_providers()[0]
        
        print(f"# ONNX Runtime provider: {self.active_provider}", file=sys.stderr)
        print(f"# Model: {model_path.name}", file=sys.stderr)
    
    def run(self, inputs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, ...]:
        """Run inference and return outputs."""
        outputs = self.session.run(None, inputs)
        return tuple(outputs)
    
    @property
    def is_gpu(self) -> bool:
        return 'CUDA' in self.active_provider


# ==============================================================================
# MCTS Implementation (ONNX-based)
# ==============================================================================

@dataclass
class MCTSSettings:
    simulations: int = 256
    c_puct: float = 1.5
    max_depth: int = 128
    max_children: int = 64
    root_dirichlet_alpha: float = 0.0
    root_exploration_frac: float = 0.0
    final_temperature: float = 0.0
    final_top_p: float = 0.90
    contempt: float = 0.15
    fpu_reduction: float = 0.20
    leaf_batch_size: int = 1
    virtual_loss: float = 1.0
    tree_reuse: bool = False


@dataclass
class MCTSResult:
    output: PolicyOutput
    root: "_Node"
    root_fen: str
    chosen_move: Optional[chess.Move]


class _Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children", "expanded", "pred_time_s")
    
    def __init__(self, prior: float) -> None:
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[chess.Move, _Node] = {}
        self.expanded = False
        self.pred_time_s = 0.0
    
    def q(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


def _terminal_value_for_side_to_move(board: chess.Board) -> Optional[float]:
    """Returns value from perspective of side to move, or None if non-terminal."""
    if not board.is_game_over(claim_draw=True):
        return None
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def _ctx_for_side_to_move(base: ContextOptions, *, root_turn: chess.Color, node_turn: chess.Color, halfmove_clock: int) -> ContextOptions:
    """Align context to the side-to-move at the node."""
    if node_turn == root_turn:
        return ContextOptions(
            active_elo=base.active_elo,
            opponent_elo=base.opponent_elo,
            active_clock_s=base.active_clock_s,
            opponent_clock_s=base.opponent_clock_s,
            active_inc_s=base.active_inc_s,
            opponent_inc_s=base.opponent_inc_s,
            halfmove_clock=halfmove_clock,
        )
    return ContextOptions(
        active_elo=base.opponent_elo,
        opponent_elo=base.active_elo,
        active_clock_s=base.opponent_clock_s,
        opponent_clock_s=base.active_clock_s,
        active_inc_s=base.opponent_inc_s,
        opponent_inc_s=base.active_inc_s,
        halfmove_clock=halfmove_clock,
    )


def _evaluate_position_onnx(
    *,
    session: OnnxInferenceSession,
    board: chess.Board,
    ctx: ContextOptions,
    root_turn: chess.Color,
    time_history_s: Optional[List[float]],
    contempt: float = 0.0,
) -> Tuple[Dict[chess.Move, float], float, float]:
    """Evaluate a position using ONNX session."""
    
    board_history, repetition_flags = _history_from_board(board)
    eff_ctx = _ctx_for_side_to_move(ctx, root_turn=root_turn, node_turn=board.turn, halfmove_clock=int(board.halfmove_clock))
    
    batch = make_model_batch_numpy(
        board=board,
        board_history=board_history,
        repetition_flags=repetition_flags,
        time_history_s=time_history_s,
        ctx=eff_ctx,
    )
    
    outputs = session.run(batch)
    move_logits = outputs[0][0]  # (4098,)
    v_cls = outputs[2][0]  # (3,) WDL
    t_logits = outputs[4][0]  # (256,)
    promo_logits = outputs[6][0]  # (8, 4)
    
    # Softmax for promo
    promo_logits = promo_logits.astype(np.float64)
    promo_p = np.exp(promo_logits - promo_logits.max(axis=-1, keepdims=True))
    promo_p = promo_p / (promo_p.sum(axis=-1, keepdims=True) + 1e-12)
    
    # WDL value [L, D, W]
    wdl = np.exp(v_cls.astype(np.float64) - v_cls.max())
    wdl = wdl / (wdl.sum() + 1e-12)
    value = float(wdl[2] - wdl[0] - contempt * wdl[1])
    
    # Expected time
    t_probs = np.exp(t_logits.astype(np.float64) - t_logits.max())
    t_probs = t_probs / (t_probs.sum() + 1e-12)
    bin_idx = np.arange(256)
    bin_seconds = ((bin_idx + 0.5) / 256.0) ** 2 * max(1e-6, eff_ctx.active_clock_s)
    exp_time_s = float(np.sum(t_probs * bin_seconds))
    
    real_turn = board.turn
    canonical = canonicalize(board)
    
    logits: List[float] = []
    moves: List[chess.Move] = []
    
    for mv in canonical.legal_moves:
        base_idx = mv.from_square * 64 + mv.to_square
        logit = float(move_logits[base_idx])
        if mv.promotion is not None and 56 <= mv.to_square <= 63:
            file_idx = mv.to_square - 56
            p_idx = PROMO_INDEX.get(mv.promotion, 0)
            promo_prob = float(promo_p[file_idx, p_idx])
            logit += float(np.log(max(1e-8, promo_prob)))
        
        real_mv = _canonical_to_real_move(mv, real_turn)
        moves.append(real_mv)
        logits.append(logit)
    
    if not moves:
        return {}, value, exp_time_s
    
    x = np.array(logits, dtype=np.float64)
    x = x - float(np.max(x))
    probs = np.exp(x)
    probs = probs / float(np.sum(probs) + 1e-12)
    
    priors: Dict[chess.Move, float] = {mv: float(p) for mv, p in zip(moves, probs)}
    return priors, value, exp_time_s


def _select_child(node: _Node, *, c_puct: float, fpu_reduction: float) -> Tuple[chess.Move, _Node]:
    assert node.children
    sqrt_n = np.sqrt(max(1, node.visit_count))
    parent_q = node.q()
    use_fpu = float(fpu_reduction) > 0.0
    
    best_move: Optional[chess.Move] = None
    best_child: Optional[_Node] = None
    best_score = -1e30
    
    for mv, child in node.children.items():
        u = c_puct * child.prior * (sqrt_n / (1 + child.visit_count))
        if use_fpu and child.visit_count == 0:
            v_term = float(parent_q - fpu_reduction)
        else:
            v_term = float(-child.q())
        
        score = v_term + u
        if score > best_score:
            best_score = score
            best_move = mv
            best_child = child
    
    assert best_move is not None and best_child is not None
    return best_move, best_child


def _apply_root_noise(
    priors: Dict[chess.Move, float],
    *,
    alpha: float,
    frac: float,
    rng: np.random.Generator,
) -> Dict[chess.Move, float]:
    if frac <= 0.0 or alpha <= 0.0 or len(priors) <= 1:
        return priors
    
    moves = list(priors.keys())
    p = np.array([priors[m] for m in moves], dtype=np.float64)
    p = p / float(p.sum() + 1e-12)
    
    noise = rng.dirichlet([alpha] * len(moves))
    mixed = (1.0 - frac) * p + frac * noise
    mixed = mixed / float(mixed.sum() + 1e-12)
    
    return {m: float(pp) for m, pp in zip(moves, mixed)}


def _prune_priors(priors: Dict[chess.Move, float], *, max_children: int) -> Dict[chess.Move, float]:
    if max_children <= 0 or len(priors) <= max_children:
        return priors
    
    items = sorted(priors.items(), key=lambda kv: kv[1], reverse=True)[:max_children]
    s = float(sum(p for _, p in items))
    if s <= 0:
        return {mv: 1.0 / len(items) for mv, _ in items}
    return {mv: float(p / s) for mv, p in items}


def find_subtree_by_move_sequence(
    last_root: Optional[_Node],
    moves_since_last_search: List[chess.Move],
) -> Optional[_Node]:
    """Find a reusable subtree by following moves from last root."""
    if last_root is None or not moves_since_last_search:
        return None
    
    node = last_root
    for move in moves_since_last_search:
        if not node.expanded or not node.children:
            return None
        if move not in node.children:
            return None
        node = node.children[move]
    
    if node.expanded:
        return node
    return None


def mcts_choose_move_onnx(
    *,
    session: OnnxInferenceSession,
    board: chess.Board,
    ctx: ContextOptions,
    time_history_s: Optional[List[float]],
    settings: MCTSSettings,
    rng: Optional[np.random.Generator] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    reuse_root: Optional[_Node] = None,
) -> MCTSResult:
    """MCTS using ONNX Runtime for evaluation."""
    
    if rng is None:
        rng = np.random.default_rng()
    
    if board.is_game_over(claim_draw=False):
        return MCTSResult(
            output=PolicyOutput(move=None, policy_prob=1.0),
            root=_Node(1.0),
            root_fen=board.fen(),
            chosen_move=None,
        )
    
    root_turn = board.turn
    root_fen = board.fen()
    
    # Check for tree reuse
    reused = False
    if reuse_root is not None and reuse_root.expanded and reuse_root.children:
        legal_moves = set(board.legal_moves)
        reuse_moves = set(reuse_root.children.keys())
        if reuse_moves & legal_moves:
            root = reuse_root
            reused = True
            root_value = root.q() if root.visit_count > 0 else 0.0
            root_time_s = root.pred_time_s
        else:
            reuse_root = None
    
    if not reused:
        root = _Node(1.0)
        
        priors, root_value, root_time_s = _evaluate_position_onnx(
            session=session,
            board=board,
            ctx=ctx,
            root_turn=root_turn,
            time_history_s=time_history_s,
            contempt=float(settings.contempt),
        )
        root.pred_time_s = float(root_time_s)
        
        # Check for immediate mate
        for mv in board.legal_moves:
            b2 = board.copy(stack=False)
            b2.push(mv)
            tv = _terminal_value_for_side_to_move(b2)
            if tv is not None and tv == -1.0:
                return MCTSResult(
                    output=PolicyOutput(move=mv, policy_prob=1.0),
                    root=root,
                    root_fen=root_fen,
                    chosen_move=mv,
                )
        
        priors = _prune_priors(priors, max_children=int(settings.max_children))
        priors = _apply_root_noise(
            priors,
            alpha=float(settings.root_dirichlet_alpha),
            frac=float(settings.root_exploration_frac),
            rng=rng,
        )
        
        for mv, p in priors.items():
            root.children[mv] = _Node(p)
        root.expanded = True
    else:
        # Check for immediate mate even when reusing
        for mv in board.legal_moves:
            b2 = board.copy(stack=False)
            b2.push(mv)
            tv = _terminal_value_for_side_to_move(b2)
            if tv is not None and tv == -1.0:
                return MCTSResult(
                    output=PolicyOutput(move=mv, policy_prob=1.0),
                    root=root,
                    root_fen=root_fen,
                    chosen_move=mv,
                )
    
    # Run simulations
    sims = max(1, int(settings.simulations))
    max_depth = max(1, int(settings.max_depth))
    vloss = float(settings.virtual_loss)
    
    done = 0
    node_count = 1 + len(root.children)
    
    while done < sims:
        if stop_check is not None and stop_check():
            break
        
        node = root
        b = board.copy(stack=True)
        th = list(time_history_s) if time_history_s is not None else None
        path: List[_Node] = [node]
        
        depth = 0
        while node.expanded and node.children and depth < max_depth:
            mv, child = _select_child(
                node,
                c_puct=float(settings.c_puct),
                fpu_reduction=float(settings.fpu_reduction),
            )
            b.push(mv)
            if th is not None:
                th = [float(getattr(node, "pred_time_s", 0.0))] + th[:-1]
            node = child
            path.append(node)
            depth += 1
            
            tv = _terminal_value_for_side_to_move(b)
            if tv is not None:
                break
        
        tv = _terminal_value_for_side_to_move(b)
        
        # Apply virtual loss
        for n in path:
            n.visit_count += 1
            n.value_sum += -vloss
        
        if tv is not None:
            leaf_value = float(tv)
        else:
            priors2, leaf_value, leaf_time_s = _evaluate_position_onnx(
                session=session,
                board=b,
                ctx=ctx,
                root_turn=root_turn,
                time_history_s=th,
                contempt=float(settings.contempt),
            )
            
            node.pred_time_s = float(leaf_time_s)
            
            priors2 = _prune_priors(priors2, max_children=int(settings.max_children))
            if not node.expanded:
                for mv2, p2 in priors2.items():
                    node.children[mv2] = _Node(p2)
                node.expanded = True
                node_count += len(priors2)
        
        # Remove virtual loss and backup
        for n in path:
            n.visit_count -= 1
            n.value_sum -= -vloss
        
        v = leaf_value
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += v
            v = -v
        
        done += 1
    
    if not root.children:
        return MCTSResult(
            output=PolicyOutput(move=None, policy_prob=1.0),
            root=root,
            root_fen=root_fen,
            chosen_move=None,
        )
    
    # Pick move from visit counts
    moves = list(root.children.keys())
    visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)
    
    children_stats = []
    for m in moves:
        child = root.children[m]
        children_stats.append({
            "move": m.uci(),
            "visits": child.visit_count,
            "q": -child.q(),
            "prior": child.prior
        })
    children_stats.sort(key=lambda x: x["visits"], reverse=True)
    
    stats = {"root_value": root_value, "children": children_stats, "mcts_nodes": node_count}
    
    total = float(visits.sum())
    if total <= 0:
        mv = moves[int(np.argmax([root.children[m].prior for m in moves]))]
        return MCTSResult(
            output=PolicyOutput(move=mv, policy_prob=1.0 / len(moves), stats=stats),
            root=root,
            root_fen=root_fen,
            chosen_move=mv,
        )
    
    if float(settings.final_temperature) <= 0.0:
        idx = int(np.argmax(visits))
        mv = moves[idx]
        return MCTSResult(
            output=PolicyOutput(move=mv, policy_prob=float(visits[idx] / total), stats=stats),
            root=root,
            root_fen=root_fen,
            chosen_move=mv,
        )
    
    t = float(settings.final_temperature)
    top_p = float(settings.final_top_p)
    
    w = np.power(visits / total, 1.0 / max(1e-6, t))
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w[w < 0.0] = 0.0
    
    w_sum = float(w.sum())
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        w = np.ones_like(visits, dtype=np.float64) / float(len(visits))
    else:
        w = w / w_sum
    
    # Apply top-p sampling
    if 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(w)[::-1]
        sorted_probs = w[sorted_indices]
        cumsum = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumsum, top_p, side='right')
        cutoff_idx = max(1, min(cutoff_idx + 1, len(w)))
        mask = np.zeros_like(w, dtype=bool)
        mask[sorted_indices[:cutoff_idx]] = True
        w = np.where(mask, w, 0.0)
        w = w / w.sum()
    
    idx = int(rng.choice(len(moves), p=w))
    mv = moves[idx]
    return MCTSResult(
        output=PolicyOutput(move=mv, policy_prob=float(w[idx]), stats=stats),
        root=root,
        root_fen=root_fen,
        chosen_move=mv,
    )


# ==============================================================================
# Engine Logic (Choose Move)
# ==============================================================================

def analyze_position_onnx(
    *,
    session: OnnxInferenceSession,
    settings: dict,
    rng: np.random.Generator,
    board: chess.Board,
    moves_uci: List[str],
    active_clock_s: float,
    opponent_clock_s: float,
    active_inc_s: float = 0.0,
    opponent_inc_s: float = 0.0,
    time_history_s: Optional[List[float]] = None,
    tc_base_s: Optional[float] = None,
    initial_fen: str = chess.STARTING_FEN,
) -> dict:
    """Analyze position and return stats dict."""
    
    _final_board, board_history, repetition_flags = build_history_from_position(chess.Board(initial_fen), moves_uci)
    
    if board.turn == settings["human_color"]:
        active_elo = settings["human_elo"]
        opp_elo = settings["engine_elo"]
    else:
        active_elo = settings["engine_elo"]
        opp_elo = settings["human_elo"]
    
    ctx = ContextOptions(
        active_elo=active_elo,
        opponent_elo=opp_elo,
        active_clock_s=active_clock_s,
        opponent_clock_s=opponent_clock_s,
        active_inc_s=float(active_inc_s),
        opponent_inc_s=float(opponent_inc_s),
        tc_base_s=tc_base_s if tc_base_s is not None else settings.get("start_clock_s"),
        halfmove_clock=int(board.halfmove_clock),
    )
    
    batch = make_model_batch_numpy(
        board=board,
        board_history=board_history,
        repetition_flags=repetition_flags,
        time_history_s=time_history_s,
        ctx=ctx,
    )
    
    outputs = session.run(batch)
    m_logits = outputs[0][0]  # (4098,)
    v_raw = float(outputs[1][0, 0])  # value scalar
    v_cls = outputs[2][0]  # WDL (3,)
    v_err = float(outputs[3][0, 0])  # error scalar
    t_logits = outputs[4][0]  # time (256,)
    promo_logits = outputs[6][0]  # (8, 4)
    
    # Effective policy with temperature
    canonical_board = canonicalize(board)
    legal_moves_data = []
    
    # Softmax promo
    promo_logits = promo_logits.astype(np.float64)
    promo_p = np.exp(promo_logits - promo_logits.max(axis=-1, keepdims=True))
    promo_p = promo_p / (promo_p.sum(axis=-1, keepdims=True) + 1e-12)
    
    for mv in canonical_board.legal_moves:
        base_idx = mv.from_square * 64 + mv.to_square
        logit = float(m_logits[base_idx])
        if mv.promotion is not None and 56 <= mv.to_square <= 63:
            file_idx = mv.to_square - 56
            p_idx = PROMO_INDEX.get(mv.promotion, 0)
            promo_prob = float(promo_p[file_idx, p_idx])
            logit += float(np.log(max(1e-8, promo_prob)))
        legal_moves_data.append({"move": mv, "logit": logit})
    
    if not legal_moves_data:
        policy_display = []
    else:
        T = max(1e-4, float(settings["temperature"]))
        logits_vec = np.array([x["logit"] for x in legal_moves_data], dtype=np.float64)
        logits_vec = logits_vec / T
        logits_vec = logits_vec - logits_vec.max()
        probs_vec = np.exp(logits_vec)
        probs_vec = probs_vec / (probs_vec.sum() + 1e-12)
        
        sorted_idx = np.argsort(probs_vec)[::-1]
        sorted_probs = probs_vec[sorted_idx]
        cumulative_probs = np.cumsum(sorted_probs)
        
        target_p = float(settings["top_p"])
        cutoff_index = int(np.searchsorted(cumulative_probs, target_p))
        cutoff_count = min(len(sorted_probs), cutoff_index + 1)
        cutoff_count = max(1, cutoff_count)
        
        kept_indices = sorted_idx[:cutoff_count]
        kept_probs = sorted_probs[:cutoff_count]
        kept_probs = kept_probs / kept_probs.sum()
        
        policy_display = []
        real_turn = board.turn
        for i, prob in zip(kept_indices, kept_probs):
            mv = legal_moves_data[int(i)]["move"]
            real_mv = _canonical_to_real_move(mv, real_turn)
            try:
                label = board.san(real_mv)
            except Exception:
                label = real_mv.uci()
            policy_display.append({"label": label, "prob": float(prob), "uci": real_mv.uci()})
    
    # WDL
    wdl = np.exp(v_cls.astype(np.float64) - v_cls.max())
    wdl = wdl / (wdl.sum() + 1e-12)
    
    # Time distribution
    T_time = max(1e-4, float(settings.get("time_temperature", 1.0)))
    t_logits_scaled = t_logits.astype(np.float64) / T_time
    t_logits_scaled = t_logits_scaled - t_logits_scaled.max()
    time_p = np.exp(t_logits_scaled)
    time_p = time_p / (time_p.sum() + 1e-12)
    
    # Raw probs for mode
    t_probs_raw = np.exp(t_logits.astype(np.float64) - t_logits.max())
    t_probs_raw = t_probs_raw / (t_probs_raw.sum() + 1e-12)
    
    mode_bin = int(np.argmax(t_probs_raw))
    mode_time_s = _time_bin_to_seconds(mode_bin, active_clock_s)
    
    # Expected time
    target_time_p = float(settings.get("time_top_p", 0.95))
    t_sorted_idx = np.argsort(time_p)[::-1]
    t_cumsum = np.cumsum(time_p[t_sorted_idx])
    t_cutoff = int(np.searchsorted(t_cumsum, target_time_p) + 1)
    t_active_idx = t_sorted_idx[:t_cutoff]
    t_active_idx.sort()
    
    expected_time_s = 0.0
    total_prob_mass = 0.0
    for idx in t_active_idx:
        sec = _time_bin_to_seconds(int(idx), active_clock_s)
        prob = float(time_p[idx])
        expected_time_s += prob * sec
        total_prob_mass += prob
    if total_prob_mass > 0:
        expected_time_s /= total_prob_mass
    
    # Sample time
    if settings.get("use_mode_time", False):
        time_sample_s = mode_time_s
        time_sample_prob = float(t_probs_raw[mode_bin])
    elif settings.get("use_expected_time", False):
        time_sample_s = expected_time_s
        time_sample_prob = 1.0
    else:
        time_sample = sample_from_logits_numpy(t_logits, temperature=T_time, top_p=target_time_p, rng=rng)
        time_sample_bin = int(time_sample.move_index)
        time_sample_s = _time_bin_to_seconds(time_sample_bin, active_clock_s)
        time_sample_prob = float(time_sample.prob)
    
    # Resign/flag from unmasked policy
    raw_policy_all = np.exp(m_logits.astype(np.float64) - m_logits.max())
    raw_policy_all = raw_policy_all / (raw_policy_all.sum() + 1e-12)
    
    win_prob = float(1.0 / (1.0 + np.exp(-float(v_raw))))
    pred_sq_error = float(v_err)
    error_bar = float(np.sqrt(max(0.0, pred_sq_error)))
    
    return {
        "top_moves": policy_display,
        "resign": float(raw_policy_all[4096]) if len(raw_policy_all) > 4096 else 0.0,
        "flag": float(raw_policy_all[4097]) if len(raw_policy_all) > 4097 else 0.0,
        "wdl": {"w": float(wdl[2]), "d": float(wdl[1]), "l": float(wdl[0])},
        "value": win_prob,
        "value_error": error_bar,
        "time_sample_s": float(time_sample_s),
        "time_sample_prob": float(time_sample_prob),
        "expected_time_s": float(expected_time_s),
        "mode_time_s": float(mode_time_s),
    }


def choose_move_onnx(
    *,
    session: OnnxInferenceSession,
    board: chess.Board,
    board_history: List[List[int]],
    repetition_flags: List[int],
    ctx: ContextOptions,
    time_history_s: Optional[List[float]],
    temperature: float,
    top_p: float,
    rng: Optional[np.random.Generator] = None,
) -> PolicyOutput:
    """Single forward pass and sample a legal move using ONNX Runtime."""
    
    real_turn = board.turn
    
    batch = make_model_batch_numpy(
        board=board,
        board_history=board_history,
        repetition_flags=repetition_flags,
        time_history_s=time_history_s,
        ctx=ctx,
    )
    
    outputs = session.run(batch)
    move_logits = outputs[0][0]  # (4098,)
    promo_logits = outputs[6][0]  # (8, 4)
    
    sample = sample_from_logits_numpy(move_logits, temperature=temperature, top_p=top_p, rng=rng)
    
    if sample.move_index == 4096:
        return PolicyOutput(move=None, policy_prob=sample.prob, is_resign=True)
    if sample.move_index == 4097:
        return PolicyOutput(move=None, policy_prob=sample.prob, is_flag=True)
    
    from_sq_c = sample.move_index // 64
    to_sq_c = sample.move_index % 64
    
    canonical = canonicalize(board)
    candidates = [mv for mv in canonical.legal_moves if mv.from_square == from_sq_c and mv.to_square == to_sq_c]
    
    if not candidates:
        mv = next(iter(canonical.legal_moves))
    else:
        promo_moves = [mv for mv in candidates if mv.promotion is not None]
        if promo_moves:
            if 56 <= to_sq_c <= 63:
                file_idx = to_sq_c - 56
                promo_char = select_promo_numpy(promo_logits[file_idx], temperature=temperature, top_p=top_p, rng=rng)
            else:
                promo_char = "q"
            
            promo_piece = PROMO_CHAR_TO_PIECE[promo_char]
            mv = chess.Move(from_sq_c, to_sq_c, promotion=promo_piece)
            if mv not in canonical.legal_moves:
                mv = promo_moves[0]
        else:
            mv = candidates[0]
    
    real_move = _canonical_to_real_move(mv, real_turn)
    return PolicyOutput(move=real_move, policy_prob=sample.prob)


def choose_engine_move_onnx(
    *,
    session: OnnxInferenceSession,
    settings: dict,
    rng: np.random.Generator,
    board: chess.Board,
    moves_uci: List[str],
    active_clock_s: float,
    opponent_clock_s: float,
    active_inc_s: float = 0.0,
    opponent_inc_s: float = 0.0,
    time_history_s: Optional[List[float]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    allow_ponder_sleep: bool = True,
    mcts_reuse_root: Optional[_Node] = None,
    mcts_reuse_moves: Optional[List[chess.Move]] = None,
) -> Tuple[PolicyOutput, dict, Optional[dict], Optional[MCTSResult]]:
    """Choose engine move using ONNX Runtime."""
    
    if board.is_game_over():
        return PolicyOutput(move=None, policy_prob=1.0), {}, None, None
    
    engine_stats = analyze_position_onnx(
        session=session,
        settings=settings,
        rng=rng,
        board=board,
        moves_uci=moves_uci,
        active_clock_s=float(active_clock_s),
        opponent_clock_s=float(opponent_clock_s),
        active_inc_s=float(active_inc_s),
        opponent_inc_s=float(opponent_inc_s),
        time_history_s=time_history_s,
        tc_base_s=settings.get("start_clock_s"),
    )
    
    engine_pred_time_s = float(engine_stats.get("time_sample_s", 0.0))
    
    use_mcts = bool(settings.get("use_mcts", False))
    current_ply = len(moves_uci)
    mcts_start_ply = int(settings.get("mcts_start_ply", 0))
    use_mcts_this_move = use_mcts and (current_ply >= mcts_start_ply)
    
    if allow_ponder_sleep and bool(settings.get("simulate_thinking_time", False)) and not use_mcts_this_move:
        if stop_check is None:
            time.sleep(engine_pred_time_s)
        else:
            deadline = time.time() + max(0.0, engine_pred_time_s)
            while time.time() < deadline:
                if stop_check():
                    break
                time.sleep(0.05)
    
    ctx = ContextOptions(
        active_elo=int(settings["engine_elo"]),
        opponent_elo=int(settings["human_elo"]),
        active_clock_s=float(active_clock_s),
        opponent_clock_s=float(opponent_clock_s),
        active_inc_s=float(active_inc_s),
        opponent_inc_s=float(opponent_inc_s),
        tc_base_s=settings.get("start_clock_s"),
        halfmove_clock=int(board.halfmove_clock),
    )
    
    mcts_stats = None
    
    if use_mcts_this_move:
        sims = int(settings.get("mcts_simulations", 128))
        cpuct = float(settings.get("mcts_c_puct", 1.5))
        
        if bool(settings.get("mcts_adaptive", False)):
            scale = float(settings.get("mcts_adaptive_scale", 100.0))
            base_sims = int(settings.get("mcts_simulations", 128))
            sims = max(16, int(scale * engine_pred_time_s))
            sim_ratio = sims / max(1, base_sims)
            cpuct_scale = math.sqrt(sim_ratio)
            cpuct = cpuct * cpuct_scale
            cpuct = max(0.1, min(8.0, cpuct))
        
        mcts_settings = MCTSSettings(
            simulations=sims,
            c_puct=cpuct,
            max_children=int(settings.get("mcts_max_children", 48)),
            max_depth=int(settings.get("mcts_max_depth", 96)),
            root_dirichlet_alpha=float(settings.get("mcts_root_dirichlet_alpha", 0.0)),
            root_exploration_frac=float(settings.get("mcts_root_exploration_frac", 0.0)),
            final_temperature=float(settings.get("mcts_final_temperature", 0.0)),
            final_top_p=float(settings.get("mcts_final_top_p", 1.0)),
            fpu_reduction=float(settings.get("mcts_fpu_reduction", 0.0)),
            contempt=float(settings.get("mcts_contempt", 0.15)),
            tree_reuse=bool(settings.get("mcts_tree_reuse", False)),
        )
        
        mcts_start_time = time.time()
        
        reuse_root: Optional[_Node] = None
        if mcts_settings.tree_reuse and mcts_reuse_root is not None and mcts_reuse_moves:
            reuse_root = find_subtree_by_move_sequence(mcts_reuse_root, mcts_reuse_moves)
        
        mcts_result = mcts_choose_move_onnx(
            session=session,
            board=board,
            ctx=ctx,
            time_history_s=time_history_s,
            settings=mcts_settings,
            rng=rng,
            stop_check=stop_check,
            reuse_root=reuse_root,
        )
        out = mcts_result.output
        mcts_stats = out.stats
        
        mcts_elapsed_s = time.time() - mcts_start_time
        
        if bool(settings.get("mcts_simulate_time", False)):
            remaining_time = engine_pred_time_s - mcts_elapsed_s
            if remaining_time > 0:
                time.sleep(remaining_time)
        
        return out, engine_stats, mcts_stats, mcts_result
    else:
        _fb, bh, rf = build_history_from_position(chess.Board(), moves_uci)
        
        opening_length = int(settings.get("opening_length", 10))
        if current_ply < opening_length:
            effective_temperature = float(settings.get("opening_temperature", settings["temperature"]))
        else:
            effective_temperature = float(settings["temperature"])
        
        out = choose_move_onnx(
            session=session,
            board=board,
            board_history=bh,
            repetition_flags=rf,
            ctx=ctx,
            time_history_s=time_history_s,
            temperature=effective_temperature,
            top_p=float(settings["top_p"]),
            rng=rng,
        )
    
    return out, engine_stats, mcts_stats, None


# ==============================================================================
# UCI Options
# ==============================================================================

@dataclass
class _Option:
    name: str
    uci_type: str
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    combo: Optional[List[str]] = None


# ==============================================================================
# UCI Engine
# ==============================================================================

class UciOnnxEngine:
    """UCI Engine using ONNX Runtime for inference."""
    
    def __init__(self, model_path: Optional[Path] = None, prefer_gpu: bool = True):
        # Find model path
        if model_path is None:
            # Look for model in same directory as this script
            script_dir = Path(__file__).parent
            model_path = script_dir / "marvin_small.onnx"
            if not model_path.exists():
                # Try current directory
                model_path = Path("marvin_small.onnx")
        
        if not model_path.exists():
            print(f"Error: ONNX model not found at {model_path}", file=sys.stderr)
            print("Run 'python scripts/export_onnx.py' to create the model.", file=sys.stderr)
            sys.exit(1)
        
        self.session = OnnxInferenceSession(model_path, prefer_gpu=prefer_gpu)
        
        seed = DEFAULT_RNG_SEED if os.environ.get("MARVIN_DETERMINISTIC") else None
        self.rng = np.random.default_rng(seed)
        
        self.settings: dict = dict(DEFAULT_GAME_SETTINGS)
        self.board = chess.Board()
        self._base_fen: str = chess.Board().fen()
        self.moves_uci: List[str] = []
        self.pred_time_s_history: List[float] = []
        
        self._search_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._search_thread: Optional[threading.Thread] = None
        self._last_bestmove: Optional[str] = None
        
        self._state_generation: int = 0
        self._active_search_id: int = 0
        
        self._has_last_go: bool = False
        self._last_go_wtime_s: float = float(START_CLOCK_S)
        self._last_go_btime_s: float = float(START_CLOCK_S)
        self._last_go_winc_s: float = 0.0
        self._last_go_binc_s: float = 0.0
        
        self._last_seen_wtime_s: float = float(START_CLOCK_S)
        self._last_seen_btime_s: float = float(START_CLOCK_S)
        
        self.internal_wtime_s: float = float(START_CLOCK_S)
        self.internal_btime_s: float = float(START_CLOCK_S)
        
        self._last_mcts_result: Optional[MCTSResult] = None
        self._last_mcts_ply: int = -1
        
        self.options = self._build_options()
    
    def _build_options(self) -> List[_Option]:
        return [
            _Option("Temperature", "string", str(self.settings["temperature"])),
            _Option("TopP", "string", str(self.settings["top_p"])),
            _Option("TimeTemperature", "string", str(self.settings["time_temperature"])),
            _Option("TimeTopP", "string", str(self.settings["time_top_p"])),
            _Option("OpeningTemperature", "string", str(self.settings.get("opening_temperature", 1.2))),
            _Option("OpeningLength", "spin", int(self.settings.get("opening_length", 10)), min=0, max=100),
            _Option("UseModeTime", "check", bool(self.settings["use_mode_time"])),
            _Option("UseExpectedTime", "check", bool(self.settings["use_expected_time"])),
            _Option("UseRealTime", "check", bool(self.settings.get("use_real_time", False))),
            _Option("HumanElo", "spin", int(self.settings["human_elo"]), min=1200, max=2400),
            _Option("EngineElo", "spin", int(self.settings["engine_elo"]), min=1200, max=2400),
            _Option("SimulateThinkingTime", "check", bool(self.settings.get("simulate_thinking_time", False))),
            _Option("InternalClock", "check", bool(self.settings.get("internal_clock", False))),
            _Option("DebugClocks", "check", bool(self.settings.get("debug_clocks", False))),
            _Option("GameBaseTime", "string", str(self.settings.get("game_base_time_s", 0))),
            _Option("GameIncrement", "string", str(self.settings.get("game_increment_s", 0))),
            _Option("LogResignProbs", "check", bool(self.settings.get("log_resign_probs", False))),
            _Option("LogTimeHistory", "check", bool(self.settings.get("log_time_history", False))),
            _Option("LogMctsStats", "check", bool(self.settings.get("log_mcts_stats", False))),
            _Option("EnableResign", "check", bool(self.settings.get("enable_resign", False))),
            _Option("ResignThreshold", "string", str(self.settings.get("resign_threshold", 0.98))),
            _Option("MinResignPly", "spin", int(self.settings.get("resign_min_ply", 20)), min=0, max=1000),
            _Option("EnableFlag", "check", bool(self.settings.get("enable_flag", False))),
            _Option("FlagThreshold", "string", str(self.settings.get("flag_threshold", 0.98))),
            _Option("UseMCTS", "check", bool(self.settings["use_mcts"])),
            _Option("MCTSSimulations", "spin", int(self.settings["mcts_simulations"]), min=1, max=200000),
            _Option("MCTSCpuct", "string", str(self.settings["mcts_c_puct"])),
            _Option("MCTSMaxChildren", "spin", int(self.settings["mcts_max_children"]), min=1, max=4096),
            _Option("MCTSRootDirichletAlpha", "string", str(self.settings["mcts_root_dirichlet_alpha"])),
            _Option("MCTSRootExplorationFrac", "string", str(self.settings["mcts_root_exploration_frac"])),
            _Option("MCTSFinalTemperature", "string", str(self.settings["mcts_final_temperature"])),
            _Option("MCTSFinalTopP", "string", str(self.settings.get("mcts_final_top_p", 1.0))),
            _Option("MCTSMaxDepth", "spin", int(self.settings["mcts_max_depth"]), min=1, max=512),
            _Option("MCTSAdaptive", "check", bool(self.settings["mcts_adaptive"])),
            _Option("MCTSAdaptiveScale", "string", str(self.settings["mcts_adaptive_scale"])),
            _Option("MCTSFPU", "string", str(self.settings.get("mcts_fpu_reduction", 0.0))),
            _Option("MCTSContempt", "string", str(self.settings.get("mcts_contempt", 0.15))),
            _Option("MCTSSimulateTime", "check", bool(self.settings.get("mcts_simulate_time", False))),
            _Option("MCTSStartPly", "spin", int(self.settings.get("mcts_start_ply", 0)), min=0, max=100),
            _Option("MCTSTreeReuse", "check", bool(self.settings.get("mcts_tree_reuse", False))),
        ]
    
    def _print(self, line: str) -> None:
        sys.stdout.write(line.rstrip("\n") + "\n")
        sys.stdout.flush()
    
    def _time_history_last8_newest_first(self) -> List[float]:
        out = list(reversed(self.pred_time_s_history[-8:]))
        while len(out) < 8:
            out.append(0.0)
        return out[:8]
    
    def _reset_position(self) -> None:
        self.board = chess.Board()
        self._base_fen = chess.Board().fen()
        self.moves_uci = []
        self.pred_time_s_history = []
        self._has_last_go = False
        self.internal_wtime_s = float(self.settings.get("start_clock_s", START_CLOCK_S))
        self.internal_btime_s = float(self.settings.get("start_clock_s", START_CLOCK_S))
        self.settings["human_color"] = (not self.board.turn)
        self._last_mcts_result = None
        self._last_mcts_ply = -1
    
    def _invalidate_search_locked(self) -> Optional[threading.Thread]:
        self._active_search_id += 1
        self._state_generation += 1
        t = self._search_thread
        if t is not None and t.is_alive():
            self._stop_event.set()
        return t
    
    def _set_position(self, *, board: chess.Board, moves: List[str]) -> None:
        base_fen = board.fen() if board.fen() != chess.Board().fen() else chess.Board().fen()
        base_board = chess.Board(fen=base_fen)
        
        if base_fen != self._base_fen:
            self._base_fen = base_fen
            self.board = base_board.copy(stack=False)
            self.moves_uci = []
            self.pred_time_s_history = []
        
        common = 0
        max_common = min(len(self.moves_uci), len(moves))
        while common < max_common and self.moves_uci[common] == moves[common]:
            common += 1
        
        if common != len(self.moves_uci) or len(moves) < len(self.moves_uci):
            new_board = base_board.copy(stack=False)
            for uci in moves:
                mv = chess.Move.from_uci(uci)
                if mv not in new_board.legal_moves:
                    raise ValueError(f"Illegal move in position: {uci}")
                new_board.push(mv)
            
            kept_times = self.pred_time_s_history[:common]
            self.board = new_board
            self.moves_uci = list(moves)
            self.pred_time_s_history = kept_times + [0.0] * (len(moves) - common)
        else:
            for uci in moves[common:]:
                mv = chess.Move.from_uci(uci)
                if mv not in self.board.legal_moves:
                    raise ValueError(f"Illegal move in position: {uci}")
                self.pred_time_s_history.append(0.0)
                self.board.push(mv)
                self.moves_uci.append(uci)
        
        self.settings["human_color"] = (not self.board.turn)
    
    def _handle_uci(self) -> None:
        self._print("id name marvin-onnx")
        self._print("id author zingalorp")
        
        for opt in self.options:
            if opt.uci_type == "check":
                default = "true" if bool(opt.default) else "false"
                self._print(f"option name {opt.name} type check default {default}")
            elif opt.uci_type == "combo":
                assert opt.combo is not None
                vars_part = " ".join([f"var {v}" for v in opt.combo])
                self._print(f"option name {opt.name} type combo default {opt.default} {vars_part}")
            elif opt.uci_type == "string":
                self._print(f"option name {opt.name} type string default {opt.default}")
            else:
                mn = int(opt.min) if opt.min is not None else 0
                mx = int(opt.max) if opt.max is not None else 100
                self._print(f"option name {opt.name} type spin default {opt.default} min {mn} max {mx}")
        
        self._print("uciok")
    
    def _handle_setoption(self, line: str) -> None:
        tokens = line.strip().split()
        if len(tokens) < 3:
            return
        
        try:
            name_idx = tokens.index("name")
        except ValueError:
            return
        
        try:
            value_idx = tokens.index("value")
        except ValueError:
            value_idx = -1
        
        if value_idx == -1:
            name = " ".join(tokens[name_idx + 1:])
            value = ""
        else:
            name = " ".join(tokens[name_idx + 1:value_idx])
            value = " ".join(tokens[value_idx + 1:])
        
        name_key = name.strip().lower()
        
        def set_setting(key: str, v: Any) -> None:
            self.settings[key] = v
        
        # Map option names to settings
        option_map = {
            "temperature": ("temperature", float),
            "topp": ("top_p", float),
            "timetemperature": ("time_temperature", float),
            "timetopp": ("time_top_p", float),
            "openingtemperature": ("opening_temperature", float),
            "openinglength": ("opening_length", lambda x: int(float(x))),
            "usemodetime": ("use_mode_time", _bool_from_uci),
            "useexpectedtime": ("use_expected_time", _bool_from_uci),
            "userealtime": ("use_real_time", _bool_from_uci),
            "humanelo": ("human_elo", lambda x: int(float(x))),
            "engineelo": ("engine_elo", lambda x: int(float(x))),
            "simulatethinkingtime": ("simulate_thinking_time", _bool_from_uci),
            "internalclock": ("internal_clock", _bool_from_uci),
            "debugclocks": ("debug_clocks", _bool_from_uci),
            "gamebasetime": ("game_base_time_s", float),
            "gameincrement": ("game_increment_s", float),
            "logresignprobs": ("log_resign_probs", _bool_from_uci),
            "logtimehistory": ("log_time_history", _bool_from_uci),
            "logmctsstats": ("log_mcts_stats", _bool_from_uci),
            "enableresign": ("enable_resign", _bool_from_uci),
            "resignthreshold": ("resign_threshold", float),
            "minresignply": ("resign_min_ply", lambda x: int(float(x))),
            "enableflag": ("enable_flag", _bool_from_uci),
            "flagthreshold": ("flag_threshold", float),
            "usemcts": ("use_mcts", _bool_from_uci),
            "mctssimulations": ("mcts_simulations", lambda x: int(float(x))),
            "mctscpuct": ("mcts_c_puct", float),
            "mctsmaxchildren": ("mcts_max_children", lambda x: int(float(x))),
            "mctsrootdirichletalpha": ("mcts_root_dirichlet_alpha", float),
            "mctsrootexplorationfrac": ("mcts_root_exploration_frac", float),
            "mctsfinaltemperature": ("mcts_final_temperature", float),
            "mctsfinaltopp": ("mcts_final_top_p", float),
            "mctsmaxdepth": ("mcts_max_depth", lambda x: int(float(x))),
            "mctsadaptive": ("mcts_adaptive", _bool_from_uci),
            "mctsadaptivescale": ("mcts_adaptive_scale", float),
            "mctsfpu": ("mcts_fpu_reduction", float),
            "mctscontempt": ("mcts_contempt", float),
            "mctssimulatetime": ("mcts_simulate_time", _bool_from_uci),
            "mctsstartply": ("mcts_start_ply", lambda x: int(float(x))),
            "mctstreereuse": ("mcts_tree_reuse", _bool_from_uci),
        }
        
        if name_key in option_map:
            key, converter = option_map[name_key]
            try:
                set_setting(key, converter(value))
            except Exception:
                pass
        
        # Handle special cases
        if name_key == "gamebasetime":
            base_s = float(value)
            if base_s > 0:
                set_setting("start_clock_s", base_s)
        
        try:
            self._print(f"info string setoption {name}={value}")
        except Exception:
            pass
    
    def _parse_position(self, line: str) -> Tuple[chess.Board, List[str]]:
        tokens = line.strip().split()
        if len(tokens) < 2:
            return chess.Board(), []
        
        idx = 1
        if tokens[idx] == "startpos":
            board = chess.Board()
            idx += 1
        elif tokens[idx] == "fen":
            idx += 1
            fen_parts: List[str] = []
            while idx < len(tokens) and tokens[idx] != "moves":
                fen_parts.append(tokens[idx])
                idx += 1
            fen = " ".join(fen_parts)
            board = chess.Board(fen=fen)
        else:
            board = chess.Board()
        
        moves: List[str] = []
        if idx < len(tokens) and tokens[idx] == "moves":
            moves = tokens[idx + 1:]
        
        return board, moves
    
    def _maybe_update_real_time_history(
        self,
        *,
        wtime_s: float,
        btime_s: float,
        winc_s: float,
        binc_s: float,
    ) -> None:
        if not bool(self.settings.get("use_real_time", False)):
            return
        if not self._has_last_go:
            return
        
        prev_w = float(self._last_go_wtime_s)
        prev_b = float(self._last_go_btime_s)
        inc_w = float(winc_s)
        inc_b = float(binc_s)
        curr_w = float(wtime_s)
        curr_b = float(btime_s)
        
        spent_w = max(0.0, prev_w + inc_w - curr_w)
        spent_b = max(0.0, prev_b + inc_b - curr_b)
        
        if len(self.pred_time_s_history) >= 1:
            last_mover = not self.board.turn
            self.pred_time_s_history[-1] = float(spent_w if last_mover == chess.WHITE else spent_b)
    
    def _search_worker(
        self,
        *,
        wtime_s: float,
        btime_s: float,
        winc_s: float,
        binc_s: float,
        search_id: int,
        state_generation: int,
    ) -> None:
        try:
            with self._search_lock:
                self._last_bestmove = None
                board = self.board.copy(stack=True)
                moves_uci = list(self.moves_uci)
                time_hist = self._time_history_last8_newest_first()
                current_ply = len(moves_uci)
                
                mcts_reuse_root: Optional[_Node] = None
                mcts_reuse_moves: List[chess.Move] = []
                
                if (
                    bool(self.settings.get("mcts_tree_reuse", False))
                    and self._last_mcts_result is not None
                    and self._last_mcts_ply >= 0
                    and self._last_mcts_result.chosen_move is not None
                ):
                    moves_since = current_ply - self._last_mcts_ply
                    if 0 < moves_since <= 2:
                        mcts_reuse_root = self._last_mcts_result.root
                        for i in range(self._last_mcts_ply, current_ply):
                            if i < len(moves_uci):
                                mcts_reuse_moves.append(chess.Move.from_uci(moves_uci[i]))
            
            if bool(self.settings.get("log_time_history", False)):
                self._print(f"info string time_history {time_hist}")
            
            if board.turn == chess.WHITE:
                active_clock_s = float(wtime_s)
                opponent_clock_s = float(btime_s)
                active_inc_s = float(winc_s)
                opponent_inc_s = float(binc_s)
            else:
                active_clock_s = float(btime_s)
                opponent_clock_s = float(wtime_s)
                active_inc_s = float(binc_s)
                opponent_inc_s = float(winc_s)
            
            out, engine_stats, _mcts_stats, mcts_result = choose_engine_move_onnx(
                session=self.session,
                settings=self.settings,
                rng=self.rng,
                board=board,
                moves_uci=moves_uci,
                active_clock_s=active_clock_s,
                opponent_clock_s=opponent_clock_s,
                active_inc_s=active_inc_s,
                opponent_inc_s=opponent_inc_s,
                time_history_s=time_hist,
                stop_check=self._stop_event.is_set,
                allow_ponder_sleep=True,
                mcts_reuse_root=mcts_reuse_root,
                mcts_reuse_moves=mcts_reuse_moves,
            )
            
            try:
                resign_p = float(engine_stats.get("resign", 0.0)) if isinstance(engine_stats, dict) else 0.0
            except Exception:
                resign_p = 0.0
            try:
                flag_p = float(engine_stats.get("flag", 0.0)) if isinstance(engine_stats, dict) else 0.0
            except Exception:
                flag_p = 0.0
            
            enable_resign = bool(self.settings.get("enable_resign", False))
            enable_flag = bool(self.settings.get("enable_flag", False))
            resign_thr = float(self.settings.get("resign_threshold", 0.98))
            flag_thr = float(self.settings.get("flag_threshold", 0.98))
            min_resign_ply = int(self.settings.get("resign_min_ply", 20))
            
            if enable_resign and resign_p >= resign_thr and current_ply >= min_resign_ply:
                self._print(f"info string action=resign resign_p={resign_p:.4f} flag_p={flag_p:.4f}")
            if enable_flag and flag_p >= flag_thr and current_ply >= min_resign_ply:
                self._print(f"info string action=flag resign_p={resign_p:.4f} flag_p={flag_p:.4f}")
            if bool(self.settings.get("log_resign_probs", False)):
                self._print(f"info string probs resign_p={resign_p:.4f} flag_p={flag_p:.4f}")
            
            try:
                if bool(self.settings.get("log_mcts_stats", False)) and isinstance(_mcts_stats, dict):
                    mn = int(_mcts_stats.get("mcts_nodes", 0))
                    tree_reused = mcts_reuse_root is not None and len(mcts_reuse_moves) > 0
                    self._print(f"info string mcts_nodes={mn} tree_reused={tree_reused}")
            except Exception:
                pass
            
            if bool(self.settings.get("log_time_history", False)):
                pred_t = float(engine_stats.get("time_sample_s", 0.0)) if isinstance(engine_stats, dict) else 0.0
                mode_t = float(engine_stats.get("mode_time_s", 0.0)) if isinstance(engine_stats, dict) else 0.0
                expected_t = float(engine_stats.get("expected_time_s", 0.0)) if isinstance(engine_stats, dict) else 0.0
                self._print(f"info string pred_time sample={pred_t:.2f}s mode={mode_t:.2f}s expected={expected_t:.2f}s")
            
            if out.move is not None:
                bestmove = out.move.uci()
            else:
                if not any(board.legal_moves):
                    bestmove = "0000"
                else:
                    fallback_uci: Optional[str] = None
                    try:
                        tm = engine_stats.get("top_moves") if isinstance(engine_stats, dict) else None
                        if tm and isinstance(tm, list):
                            u = tm[0].get("uci")
                            if isinstance(u, str) and u:
                                fallback_uci = u
                    except Exception:
                        fallback_uci = None
                    
                    if fallback_uci is not None:
                        mv = chess.Move.from_uci(fallback_uci)
                        if mv in board.legal_moves:
                            bestmove = fallback_uci
                        else:
                            bestmove = next(iter(board.legal_moves)).uci()
                    else:
                        bestmove = next(iter(board.legal_moves)).uci()
            
            with self._search_lock:
                if search_id != self._active_search_id or state_generation != self._state_generation:
                    return
                
                self._last_bestmove = bestmove
                
                if mcts_result is not None:
                    self._last_mcts_result = mcts_result
                    self._last_mcts_ply = len(moves_uci)
                
                if self.moves_uci == moves_uci and self.board.fen() == board.fen():
                    pred_t = float(engine_stats.get("time_sample_s", 0.0))
                    
                    if self.settings.get("internal_clock", False):
                        if board.turn == chess.WHITE:
                            self.internal_wtime_s = max(0.0, self.internal_wtime_s - pred_t + active_inc_s)
                        else:
                            self.internal_btime_s = max(0.0, self.internal_btime_s - pred_t + active_inc_s)
                    
                    if out.move is not None:
                        self.board.push(out.move)
                        self.moves_uci.append(bestmove)
                    self.pred_time_s_history.append(pred_t)
            
            self._print(f"bestmove {bestmove}")
        finally:
            with self._search_lock:
                self._search_thread = None
                self._stop_event.clear()
    
    def _handle_go(self, line: str) -> None:
        tokens = line.strip().split()
        
        args: dict = {}
        i = 1
        while i < len(tokens):
            t = tokens[i]
            if t in ("ponder", "infinite"):
                i += 1
                continue
            if t in ("wtime", "btime", "winc", "binc", "movestogo", "depth", "nodes", "mate", "movetime"):
                if i + 1 < len(tokens):
                    args[t] = tokens[i + 1]
                    i += 2
                    continue
            i += 1
        
        default_clock_s = float(self.settings.get("game_base_time_s", 0))
        if default_clock_s <= 0:
            default_clock_s = float(self.settings.get("start_clock_s", START_CLOCK_S))
        default_clock_ms = int(default_clock_s * 1000)
        
        wtime_ms = float(args.get("wtime", str(default_clock_ms)))
        btime_ms = float(args.get("btime", str(default_clock_ms)))
        wtime_s = wtime_ms / 1000.0
        btime_s = btime_ms / 1000.0
        
        default_inc_s = float(self.settings.get("game_increment_s", 0))
        default_inc_ms = int(default_inc_s * 1000)
        
        winc_s = float(args.get("winc", str(default_inc_ms))) / 1000.0
        binc_s = float(args.get("binc", str(default_inc_ms))) / 1000.0
        
        if not self._has_last_go:
            explicit_base = float(self.settings.get("game_base_time_s", 0))
            if explicit_base <= 0:
                self.settings["start_clock_s"] = float(max(wtime_s, btime_s))
        
        if self._has_last_go:
            if wtime_s <= 0.0 and self._last_seen_wtime_s > 0.0:
                wtime_s = float(self._last_seen_wtime_s)
            if btime_s <= 0.0 and self._last_seen_btime_s > 0.0:
                btime_s = float(self._last_seen_btime_s)
        
        self._last_seen_wtime_s = float(wtime_s)
        self._last_seen_btime_s = float(btime_s)
        
        if self.settings.get("internal_clock", False):
            if not self._has_last_go:
                self.internal_wtime_s = wtime_s
                self.internal_btime_s = btime_s
            wtime_s = self.internal_wtime_s
            btime_s = self.internal_btime_s
        
        with self._search_lock:
            self._maybe_update_real_time_history(wtime_s=wtime_s, btime_s=btime_s, winc_s=winc_s, binc_s=binc_s)
            self._has_last_go = True
            self._last_go_wtime_s = float(wtime_s)
            self._last_go_btime_s = float(btime_s)
            self._last_go_winc_s = float(winc_s)
            self._last_go_binc_s = float(binc_s)
        
        prev_t: Optional[threading.Thread] = None
        with self._search_lock:
            if self._search_thread is not None and self._search_thread.is_alive():
                prev_t = self._invalidate_search_locked()
        
        if prev_t is not None:
            prev_t.join(timeout=2.0)
        
        with self._search_lock:
            if self._search_thread is not None and self._search_thread.is_alive():
                return
            
            self._stop_event.clear()
            self._active_search_id += 1
            search_id = int(self._active_search_id)
            state_generation = int(self._state_generation)
            
            t = threading.Thread(
                target=self._search_worker,
                kwargs={
                    "wtime_s": wtime_s,
                    "btime_s": btime_s,
                    "winc_s": float(winc_s),
                    "binc_s": float(binc_s),
                    "search_id": search_id,
                    "state_generation": state_generation,
                },
                daemon=True,
            )
            self._search_thread = t
            t.start()
    
    def _handle_stop(self) -> None:
        with self._search_lock:
            t = self._search_thread
            if t is None:
                return
            self._stop_event.set()
        
        t.join(timeout=30.0)
    
    def loop(self) -> None:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            
            if line == "uci":
                self._handle_uci()
            elif line == "isready":
                self._print("readyok")
            elif line.startswith("setoption"):
                self._handle_setoption(line)
            elif line == "ucinewgame":
                t: Optional[threading.Thread] = None
                with self._search_lock:
                    t = self._invalidate_search_locked()
                    self._reset_position()
                if t is not None:
                    t.join(timeout=5.0)
            elif line.startswith("position"):
                board, moves = self._parse_position(line)
                t2: Optional[threading.Thread] = None
                with self._search_lock:
                    t2 = self._invalidate_search_locked()
                if t2 is not None:
                    t2.join(timeout=5.0)
                with self._search_lock:
                    self._set_position(board=board, moves=moves)
            elif line.startswith("go"):
                self._handle_go(line)
            elif line == "stop":
                self._handle_stop()
            elif line == "quit":
                self._handle_stop()
                return


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="Marvin UCI Engine (ONNX Runtime)")
    parser.add_argument("--model", "-m", type=Path, default=None, help="Path to ONNX model file")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution (disable GPU)")
    args = parser.parse_args()
    
    engine = UciOnnxEngine(model_path=args.model, prefer_gpu=not args.cpu)
    engine.loop()


if __name__ == "__main__":
    main()
