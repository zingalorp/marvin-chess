#!/usr/bin/env python3
"""
ONNX UCI Engine for Marvin Chess AI

A standalone, easy-to-use UCI engine that uses ONNX Runtime for inference.
No PyTorch required at runtime - just onnxruntime and numpy.

Usage:
    python -m inference.uci_onnx
    
Or configure in your chess GUI pointing to this script.

Requirements:
    pip install onnxruntime chess numpy
    
For GPU acceleration:
    pip install onnxruntime-gpu
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import chess

# Suppress ONNX Runtime CUDA warnings when falling back to CPU
# Set before importing onnxruntime
os.environ.setdefault("ORT_DISABLE_LOGGING", "1")

# Handle PyInstaller bundled executable
# Add the bundled DLL path to the search path
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Running as PyInstaller bundle
    bundle_dir = Path(sys._MEIPASS)
    
    # Add onnxruntime capi folder to PATH and DLL search path (Windows)
    if sys.platform == 'win32':
        import ctypes
        
        ort_capi = bundle_dir / 'onnxruntime' / 'capi'
        
        # Add bundle dir for VC++ runtime DLLs - needs to be first in PATH
        os.environ['PATH'] = str(bundle_dir) + os.pathsep + str(ort_capi) + os.pathsep + os.environ.get('PATH', '')
        
        try:
            os.add_dll_directory(str(bundle_dir))
        except (AttributeError, OSError):
            pass
        
        if ort_capi.exists():
            try:
                os.add_dll_directory(str(ort_capi))
            except (AttributeError, OSError):
                pass
            
            # Preload onnxruntime DLLs from bundle_dir (where build script copies them)
            try:
                ort_dll_root = bundle_dir / 'onnxruntime.dll'
                providers_dll_root = bundle_dir / 'onnxruntime_providers_shared.dll'
                
                if ort_dll_root.exists():
                    ctypes.windll.kernel32.LoadLibraryW(str(ort_dll_root))
                if providers_dll_root.exists():
                    ctypes.windll.kernel32.LoadLibraryW(str(providers_dll_root))
            except Exception:
                pass  # Silently continue - import will fail if DLLs not found

try:
    import onnxruntime as ort
    # Suppress provider registration warnings
    ort.set_default_logger_severity(3)  # ERROR level only
except ImportError as e:
    # More detailed error for debugging PyInstaller issues
    print(f"Error: onnxruntime not installed. Install with: pip install onnxruntime", file=sys.stderr)
    print(f"For GPU support: pip install onnxruntime-gpu", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error loading onnxruntime: {e}", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# Constants (must match training)
# ============================================================================

HISTORY_LEN = 8
NUM_SQUARES = 64
NUM_POLICY_OUTPUTS = 4098
NUM_TIME_BINS = 256

PIECE_MAP = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

TC_BLITZ = 0
TC_RAPID = 1
TC_CLASSICAL = 2

PROMO_INDEX_TO_CHAR = {0: "q", 1: "r", 2: "b", 3: "n"}
PROMO_CHAR_TO_PIECE = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}

# Default settings
DEFAULT_SETTINGS = {
    # Core sampling
    "temperature": 0.35,
    "top_p": 0.9,
    "time_temperature": 1.0,
    "time_top_p": 0.9,
    "opening_temperature": 1.2,
    "opening_length": 10,
    
    # Elo settings
    "engine_elo": 1900,
    "human_elo": 1900,
    
    # Clock/time settings
    "start_clock_s": 300.0,
    "game_base_time_s": 0,
    "game_increment_s": 0,
    
    # Time prediction modes
    "use_mode_time": True,
    "use_expected_time": False,
    "use_real_time": False,
    
    # Clock tracking
    "internal_clock": False,
    "debug_clocks": False,
    
    # Thinking time simulation
    "simulate_thinking_time": False,
    
    # Logging
    "log_time_history": False,
    "debug": False,
}

ENGINE_NAME = "marvin-onnx"
ENGINE_AUTHOR = "zingalorp"
ENGINE_VERSION = "1.0.0"


# ============================================================================
# Time Prediction Helpers
# ============================================================================

def time_bin_to_seconds(bin_idx: int, active_clock_s: float) -> float:
    """Convert a time bin index to seconds."""
    scaled_mid = (bin_idx + 0.5) / float(NUM_TIME_BINS)
    return float((scaled_mid ** 2) * max(1e-6, active_clock_s))


def sample_time_bin(
    time_logits: np.ndarray,
    temperature: float,
    top_p: float,
    rng: np.random.Generator,
) -> Tuple[int, float]:
    """Sample a time bin from logits, return (bin_idx, probability)."""
    # Apply temperature
    if temperature > 0:
        scaled = time_logits / temperature
    else:
        scaled = time_logits
    
    # Softmax
    exp_logits = np.exp(scaled - np.max(scaled))
    probs = exp_logits / np.sum(exp_logits)
    
    # Top-p (nucleus) sampling
    sorted_indices = np.argsort(probs)[::-1]
    cumsum = np.cumsum(probs[sorted_indices])
    cutoff = np.searchsorted(cumsum, top_p) + 1
    cutoff = min(cutoff, len(probs))
    
    active_indices = sorted_indices[:cutoff]
    active_probs = probs[active_indices]
    active_probs = active_probs / active_probs.sum()
    
    chosen_local = rng.choice(len(active_indices), p=active_probs)
    chosen_bin = int(active_indices[chosen_local])
    chosen_prob = float(probs[chosen_bin])
    
    return chosen_bin, chosen_prob


# ============================================================================
# Board Encoding (matches inference/encoding.py)
# ============================================================================

def canonicalize(board: chess.Board) -> chess.Board:
    """Flip board so current player is always white."""
    return board if board.turn == chess.WHITE else board.mirror()


def encode_board(board: chess.Board) -> List[int]:
    """Encode board to piece tokens (0-12)."""
    tokens = [0] * NUM_SQUARES
    for square in range(NUM_SQUARES):
        piece = board.piece_at(square)
        if piece is None:
            continue
        val = PIECE_MAP[piece.piece_type]
        tokens[square] = val if piece.color == chess.WHITE else val + 6
    return tokens


def get_tc_category(base_seconds: float, inc_seconds: float) -> int:
    """Get time control category: 0=blitz, 1=rapid, 2=classical."""
    duration = float(base_seconds) + 40.0 * float(inc_seconds)
    if duration < 600.0:
        return TC_BLITZ
    if duration < 1800.0:
        return TC_RAPID
    return TC_CLASSICAL


def build_history_from_moves(
    start_board: chess.Board,
    moves_uci: List[str],
) -> Tuple[chess.Board, List[List[int]], List[int]]:
    """Build board history and repetition flags from move list."""
    board = start_board.copy(stack=False)
    
    hist: List[List[int]] = []
    positions: List[str] = []
    
    # Initial position
    canonical = canonicalize(board)
    hist.append(encode_board(canonical))
    positions.append(canonical.board_fen())
    
    for uci in moves_uci:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            raise ValueError(f"Illegal move: {uci}")
        board.push(mv)
        canonical = canonicalize(board)
        hist.append(encode_board(canonical))
        positions.append(canonical.board_fen())
    
    # Take last HISTORY_LEN, newest first
    hist = list(reversed(hist[-HISTORY_LEN:]))
    positions = list(reversed(positions[-HISTORY_LEN:]))
    
    # Pad if needed
    while len(hist) < HISTORY_LEN:
        hist.append([0] * NUM_SQUARES)
    while len(positions) < HISTORY_LEN:
        positions.append("")
    
    # Compute repetition flags
    seen: dict[str, int] = {}
    rep_flags: List[int] = []
    for pos in positions:
        if pos:
            seen[pos] = seen.get(pos, 0) + 1
            rep_flags.append(1 if seen[pos] >= 2 else 0)
        else:
            rep_flags.append(0)
    
    return board, hist[:HISTORY_LEN], rep_flags[:HISTORY_LEN]


def make_batch_numpy(
    board: chess.Board,
    board_history: List[List[int]],
    repetition_flags: List[int],
    active_clock_s: float,
    opponent_clock_s: float,
    active_inc_s: float = 0.0,
    opponent_inc_s: float = 0.0,
    active_elo: int = 1900,
    opponent_elo: int = 1900,
    tc_base_s: Optional[float] = None,
    time_history_s: Optional[List[float]] = None,
) -> dict[str, np.ndarray]:
    """Create model input batch as numpy arrays."""
    
    if time_history_s is None:
        time_history_s = [0.0] * HISTORY_LEN
    
    # Legal move mask
    canonical = canonicalize(board)
    legal_mask = np.zeros(NUM_POLICY_OUTPUTS, dtype=np.bool_)
    for mv in canonical.legal_moves:
        idx = mv.from_square * 64 + mv.to_square
        legal_mask[idx] = True
    
    # Scalars (normalized as in training)
    active_elo_norm = (active_elo - 1900) / 700.0
    opp_elo_norm = (opponent_elo - 1900) / 700.0
    ply = board.fullmove_number * 2 - (0 if board.turn == chess.WHITE else 1)
    ply_norm = ply / 100.0
    active_clock_norm = math.log1p(max(0.0, active_clock_s)) / 10.0
    opp_clock_norm = math.log1p(max(0.0, opponent_clock_s)) / 10.0
    active_inc_norm = active_inc_s / 30.0
    opp_inc_norm = opponent_inc_s / 30.0
    hmc_norm = float(board.halfmove_clock) / 100.0
    
    scalars = np.array([
        active_elo_norm,
        opp_elo_norm,
        ply_norm,
        active_clock_norm,
        opp_clock_norm,
        active_inc_norm,
        opp_inc_norm,
        hmc_norm,
    ], dtype=np.float32)
    
    # Time control category
    base_s = tc_base_s if tc_base_s is not None else max(active_clock_s, opponent_clock_s)
    tc_cat = get_tc_category(base_s, active_inc_s)
    
    # Castling rights
    castling = np.array([
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
    ], dtype=np.float32)
    
    # En passant
    ep_mask = np.zeros(64, dtype=np.float32)
    if board.ep_square is not None:
        ep_mask[int(board.ep_square)] = 1.0
    
    # Build batch with batch dimension
    return {
        "board_history": np.array(board_history, dtype=np.int64)[np.newaxis, ...],  # (1, 8, 64)
        "time_history": (np.array(time_history_s, dtype=np.float32) / 60.0)[np.newaxis, ...],  # (1, 8)
        "rep_flags": np.array(repetition_flags, dtype=np.float32)[np.newaxis, ...],  # (1, 8)
        "castling": castling[np.newaxis, ...],  # (1, 4)
        "ep_mask": ep_mask[np.newaxis, ...],  # (1, 64)
        "scalars": scalars[np.newaxis, ...],  # (1, 8)
        "tc_cat": np.array([tc_cat], dtype=np.int64),  # (1,)
        "legal_mask": legal_mask[np.newaxis, ...],  # (1, 4098)
    }


# ============================================================================
# Sampling
# ============================================================================

def top_p_filter(probs: np.ndarray, top_p: float) -> np.ndarray:
    """Apply nucleus (top-p) sampling filter."""
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


def sample_move(
    logits: np.ndarray,
    temperature: float,
    top_p: float,
    rng: np.random.Generator,
) -> Tuple[int, float]:
    """Sample move index from logits. Returns (index, probability)."""
    if temperature <= 0.0:
        idx = int(np.argmax(logits))
        return idx, 1.0
    
    x = logits / float(temperature)
    x = x - x.max()
    probs = np.exp(x)
    probs = probs / (probs.sum() + 1e-12)
    
    probs = top_p_filter(probs, float(top_p))
    idx = int(rng.choice(len(probs), p=probs))
    return idx, float(probs[idx])


# ============================================================================
# ONNX Model Wrapper
# ============================================================================

def _get_base_path() -> Path:
    """Get the base path for finding model files.
    
    Handles both normal Python execution and PyInstaller bundles.
    """
    # PyInstaller sets sys._MEIPASS to the temp folder where files are extracted
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS)
    # Normal execution - look relative to this file
    return Path(__file__).parent


class ONNXModel:
    """Lightweight ONNX model wrapper."""
    
    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            # Try multiple locations
            base_path = _get_base_path()
            candidates = [
                base_path / "marvin_small.onnx",
                Path.cwd() / "marvin_small.onnx",
                Path(__file__).parent / "marvin_small.onnx",
            ]
            for candidate in candidates:
                if candidate.exists():
                    model_path = candidate
                    break
            else:
                model_path = candidates[0]  # For error message
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.model_path}\n"
                f"Run 'python scripts/export_onnx.py' to create it."
            )
        
        # Check for external data file
        data_path = self.model_path.with_suffix(".onnx.data")
        if not data_path.exists():
            print(f"# Warning: External data file not found: {data_path}", file=sys.stderr)
        
        # Configure providers - only request CUDA if we can actually use it
        # This avoids noisy error messages when CUDA DLLs are missing
        providers = []
        
        # Check if CUDA is actually usable (not just listed)
        try:
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                # Try to verify CUDA is actually working
                import ctypes
                try:
                    if sys.platform == 'win32':
                        ctypes.CDLL('nvcuda.dll')
                    providers.append('CUDAExecutionProvider')
                except OSError:
                    pass  # CUDA driver not available
        except Exception:
            pass
        
        providers.append('CPUExecutionProvider')
        
        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        sess_options.log_severity_level = 3  # ERROR only
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options,
            providers=providers,
        )
        
        self.provider = self.session.get_providers()[0] if self.session.get_providers() else "Unknown"
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
    
    def predict(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference and return outputs as dict."""
        outputs = self.session.run(self.output_names, batch)
        return {name: out for name, out in zip(self.output_names, outputs)}


# ============================================================================
# UCI Option
# ============================================================================

@dataclass
class UCIOption:
    name: str
    uci_type: str
    default: Any
    min_val: Optional[float] = None
    max_val: Optional[float] = None


# ============================================================================
# UCI Engine
# ============================================================================

class UCIOnnxEngine:
    """ONNX-based UCI chess engine."""
    
    def __init__(self, model_path: Optional[Path] = None):
        print(f"# Loading ONNX model...", file=sys.stderr)
        self.model = ONNXModel(model_path)
        print(f"# marvin-onnx ready (ONNX Runtime: {self.model.provider})", file=sys.stderr)
        
        self.rng = np.random.default_rng()
        self.settings = dict(DEFAULT_SETTINGS)
        
        self.board = chess.Board()
        self.moves_uci: List[str] = []
        self._base_fen = chess.STARTING_FEN
        
        self._search_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._search_lock = threading.Lock()
        
        # Internal clock tracking
        self.internal_wtime_s = 0.0
        self.internal_btime_s = 0.0
        self.pred_time_s_history: List[float] = []
        self._has_last_go = False
        self._last_go_wtime_s = 0.0
        self._last_go_btime_s = 0.0
        self._last_go_winc_s = 0.0
        self._last_go_binc_s = 0.0
        self._last_predicted_time_s = 0.0  # Cache last prediction for opponent moves
        
        self.options = self._build_options()
    
    def _build_options(self) -> List[UCIOption]:
        return [
            # Core sampling parameters
            UCIOption("Temperature", "string", str(self.settings["temperature"])),
            UCIOption("TopP", "string", str(self.settings["top_p"])),
            UCIOption("TimeTemperature", "string", str(self.settings["time_temperature"])),
            UCIOption("TimeTopP", "string", str(self.settings["time_top_p"])),
            
            # Opening variation
            UCIOption("OpeningTemperature", "string", str(self.settings["opening_temperature"])),
            UCIOption("OpeningLength", "spin", int(self.settings["opening_length"]), 0, 100),
            
            # Time prediction modes
            UCIOption("UseModeTime", "check", self.settings["use_mode_time"]),
            UCIOption("UseExpectedTime", "check", self.settings["use_expected_time"]),
            UCIOption("UseRealTime", "check", self.settings["use_real_time"]),
            
            # Elo settings (affects play style)
            UCIOption("EngineElo", "spin", int(self.settings["engine_elo"]), 100, 4000),
            UCIOption("HumanElo", "spin", int(self.settings["human_elo"]), 100, 4000),
            
            # Clock tracking
            UCIOption("InternalClock", "check", self.settings["internal_clock"]),
            UCIOption("DebugClocks", "check", self.settings["debug_clocks"]),
            
            # Time control hints (helps model understand game phase)
            UCIOption("GameBaseTime", "string", str(self.settings["game_base_time_s"])),
            UCIOption("GameIncrement", "string", str(self.settings["game_increment_s"])),
            
            # Logging options
            UCIOption("LogTimeHistory", "check", self.settings["log_time_history"]),
            
            # Behavior options
            UCIOption("SimulateThinkingTime", "check", self.settings["simulate_thinking_time"]),
            UCIOption("Debug", "check", self.settings["debug"]),
        ]
    
    def _print(self, line: str) -> None:
        sys.stdout.write(line.rstrip("\n") + "\n")
        sys.stdout.flush()
    
    def _time_history_last8_newest_first(self) -> List[float]:
        """Get the last 8 predicted times in newest-first order, padded with 0.0."""
        history = list(self.pred_time_s_history)
        if len(history) > HISTORY_LEN:
            history = history[-HISTORY_LEN:]
        # Reverse to get newest first
        history = list(reversed(history))
        # Pad with zeros if needed
        while len(history) < HISTORY_LEN:
            history.append(0.0)
        return history
    
    def _handle_uci(self) -> None:
        self._print(f"id name {ENGINE_NAME} {ENGINE_VERSION}")
        self._print(f"id author {ENGINE_AUTHOR}")
        
        for opt in self.options:
            if opt.uci_type == "check":
                default = "true" if bool(opt.default) else "false"
                self._print(f"option name {opt.name} type check default {default}")
            elif opt.uci_type == "string":
                self._print(f"option name {opt.name} type string default {opt.default}")
            else:  # spin
                mn = int(opt.min_val) if opt.min_val is not None else 0
                mx = int(opt.max_val) if opt.max_val is not None else 100
                self._print(f"option name {opt.name} type spin default {opt.default} min {mn} max {mx}")
        
        self._print("uciok")
    
    def _handle_setoption(self, line: str) -> None:
        tokens = line.strip().split()
        if len(tokens) < 3:
            return
        
        try:
            name_idx = tokens.index("name")
            value_idx = tokens.index("value") if "value" in tokens else -1
        except ValueError:
            return
        
        if value_idx == -1:
            name = " ".join(tokens[name_idx + 1:])
            value = ""
        else:
            name = " ".join(tokens[name_idx + 1:value_idx])
            value = " ".join(tokens[value_idx + 1:])
        
        name_key = name.strip().lower()
        
        def _bool(v: str) -> bool:
            return v.strip().lower() in ("1", "true", "yes", "on")
        
        if name_key == "temperature":
            self.settings["temperature"] = float(value)
        elif name_key == "topp":
            self.settings["top_p"] = float(value)
        elif name_key == "timetemperature":
            self.settings["time_temperature"] = float(value)
        elif name_key == "timetopp":
            self.settings["time_top_p"] = float(value)
        elif name_key == "openingtemperature":
            self.settings["opening_temperature"] = float(value)
        elif name_key == "openinglength":
            self.settings["opening_length"] = int(float(value))
        elif name_key == "usemodetime":
            self.settings["use_mode_time"] = _bool(value)
        elif name_key == "useexpectedtime":
            self.settings["use_expected_time"] = _bool(value)
        elif name_key == "userealtime":
            self.settings["use_real_time"] = _bool(value)
        elif name_key == "engineelo":
            self.settings["engine_elo"] = int(float(value))
        elif name_key == "humanelo":
            self.settings["human_elo"] = int(float(value))
        elif name_key == "internalclock":
            self.settings["internal_clock"] = _bool(value)
        elif name_key == "debugclocks":
            self.settings["debug_clocks"] = _bool(value)
        elif name_key == "gamebasetime":
            self.settings["game_base_time_s"] = float(value)
            if float(value) > 0:
                self.settings["start_clock_s"] = float(value)
        elif name_key == "gameincrement":
            self.settings["game_increment_s"] = float(value)
        elif name_key == "logtimehistory":
            self.settings["log_time_history"] = _bool(value)
        elif name_key == "simulatethinkingtime":
            self.settings["simulate_thinking_time"] = _bool(value)
        elif name_key == "debug":
            self.settings["debug"] = _bool(value)
        
        # Echo back for verification
        if self.settings.get("debug", False):
            self._print(f"info string setoption {name}={value}")
    
    def _handle_isready(self) -> None:
        self._print("readyok")
    
    def _handle_ucinewgame(self) -> None:
        self.board = chess.Board()
        self.moves_uci = []
        self._base_fen = chess.STARTING_FEN
        
        # Reset internal clock state
        self.internal_wtime_s = 0.0
        self.internal_btime_s = 0.0
        self.pred_time_s_history = []
        self._has_last_go = False
        self._last_go_wtime_s = 0.0
        self._last_go_btime_s = 0.0
        self._last_go_winc_s = 0.0
        self._last_go_binc_s = 0.0
        self._last_predicted_time_s = 0.0
    
    def _handle_position(self, line: str) -> None:
        tokens = line.strip().split()
        idx = 0
        
        # Skip "position"
        if tokens and tokens[0] == "position":
            idx += 1
        
        # Parse startpos or fen
        if idx < len(tokens) and tokens[idx] == "startpos":
            self.board = chess.Board()
            self._base_fen = chess.STARTING_FEN
            idx += 1
        elif idx < len(tokens) and tokens[idx] == "fen":
            idx += 1
            fen_parts = []
            while idx < len(tokens) and tokens[idx] != "moves":
                fen_parts.append(tokens[idx])
                idx += 1
            fen = " ".join(fen_parts)
            self.board = chess.Board(fen)
            self._base_fen = fen
        
        # Parse moves - track where we are vs where we were
        old_moves = list(self.moves_uci)
        self.moves_uci = []
        
        if idx < len(tokens) and tokens[idx] == "moves":
            idx += 1
            self.board = chess.Board(self._base_fen)
            
            # Find common prefix with old moves
            common = 0
            new_moves = []
            while idx < len(tokens):
                new_moves.append(tokens[idx])
                idx += 1
            
            while common < len(old_moves) and common < len(new_moves):
                if old_moves[common] != new_moves[common]:
                    break
                common += 1
            
            # If we're going backwards or starting fresh, reset time history
            if common < len(old_moves):
                self.pred_time_s_history = self.pred_time_s_history[:common]
            
            # Apply moves
            idx = 0
            while idx < len(new_moves):
                uci = new_moves[idx]
                try:
                    mv = chess.Move.from_uci(uci)
                    if mv not in self.board.legal_moves:
                        break
                    
                    # If this is a new move (beyond common prefix), handle InternalClock
                    if idx >= common and self.settings.get("internal_clock", False):
                        # Use cached time prediction from last search instead of running inference
                        # This assumes opponent takes similar time to engine's last move
                        est_t = self._last_predicted_time_s if self._last_predicted_time_s > 0 else 3.0
                        
                        # Get increment for the side that made this move
                        move_inc = self._last_go_winc_s if self.board.turn == chess.WHITE else self._last_go_binc_s
                        
                        # Update internal clock for the side that made this move
                        if self.board.turn == chess.WHITE:
                            self.internal_wtime_s = max(0.0, self.internal_wtime_s - est_t + move_inc)
                        else:
                            self.internal_btime_s = max(0.0, self.internal_btime_s - est_t + move_inc)
                        
                        self.pred_time_s_history.append(est_t)
                    elif idx >= common:
                        # Not using internal clock, but still track that we have a new move
                        self.pred_time_s_history.append(0.0)
                    
                    self.board.push(mv)
                    self.moves_uci.append(uci)
                except:
                    pass
                idx += 1
    
    def _handle_go(self, line: str) -> None:
        tokens = line.strip().split()
        
        # Parse time controls
        wtime_ms = 300000
        btime_ms = 300000
        winc_ms = 0
        binc_ms = 0
        
        for i, tok in enumerate(tokens):
            if tok == "wtime" and i + 1 < len(tokens):
                wtime_ms = int(tokens[i + 1])
            elif tok == "btime" and i + 1 < len(tokens):
                btime_ms = int(tokens[i + 1])
            elif tok == "winc" and i + 1 < len(tokens):
                winc_ms = int(tokens[i + 1])
            elif tok == "binc" and i + 1 < len(tokens):
                binc_ms = int(tokens[i + 1])
        
        # Convert to seconds
        wtime_s = wtime_ms / 1000.0
        btime_s = btime_ms / 1000.0
        winc_s = winc_ms / 1000.0
        binc_s = binc_ms / 1000.0
        
        # InternalClock mode: sync on first go, then use internal clocks
        if self.settings.get("internal_clock", False):
            if not self._has_last_go:
                # First go command - sync internal clocks
                self.internal_wtime_s = wtime_s
                self.internal_btime_s = btime_s
            
            # Use internal clocks instead of UCI clocks
            wtime_s = self.internal_wtime_s
            btime_s = self.internal_btime_s
        
        # Store for next time (used by InternalClock in position handler)
        self._has_last_go = True
        self._last_go_wtime_s = wtime_s
        self._last_go_btime_s = btime_s
        self._last_go_winc_s = winc_s
        self._last_go_binc_s = binc_s
        
        # Determine active/opponent clocks
        if self.board.turn == chess.WHITE:
            active_clock_s = wtime_s
            opponent_clock_s = btime_s
            active_inc_s = winc_s
            opponent_inc_s = binc_s
        else:
            active_clock_s = btime_s
            opponent_clock_s = wtime_s
            active_inc_s = binc_s
            opponent_inc_s = winc_s
        
        # Debug clock output
        if self.settings.get("debug_clocks", False):
            self._print(f"info string clocks before_search w={self.internal_wtime_s:.2f}s b={self.internal_btime_s:.2f}s using_internal={self.settings.get('internal_clock', False)}")
        
        # Run search in background thread
        self._stop_event.clear()
        
        def search():
            move = self._search(active_clock_s, opponent_clock_s, active_inc_s, opponent_inc_s)
            if move:
                self._print(f"bestmove {move.uci()}")
            else:
                # No legal moves
                legal = list(self.board.legal_moves)
                if legal:
                    self._print(f"bestmove {legal[0].uci()}")
                else:
                    self._print("bestmove 0000")
        
        with self._search_lock:
            self._search_thread = threading.Thread(target=search, daemon=True)
            self._search_thread.start()
    
    def _search(
        self,
        active_clock_s: float,
        opponent_clock_s: float,
        active_inc_s: float,
        opponent_inc_s: float,
    ) -> Optional[chess.Move]:
        """Run model inference and select a move."""
        
        if self.board.is_game_over():
            return None
        
        # Build history
        start_board = chess.Board(self._base_fen)
        _, board_history, rep_flags = build_history_from_moves(start_board, self.moves_uci)
        
        # Get time history
        time_history_s = self._time_history_last8_newest_first()
        
        # Create batch
        batch = make_batch_numpy(
            board=self.board,
            board_history=board_history,
            repetition_flags=rep_flags,
            active_clock_s=active_clock_s,
            opponent_clock_s=opponent_clock_s,
            active_inc_s=active_inc_s,
            opponent_inc_s=opponent_inc_s,
            active_elo=self.settings["engine_elo"],
            opponent_elo=self.settings["human_elo"],
            tc_base_s=self.settings.get("start_clock_s"),
            time_history_s=time_history_s,
        )
        
        # Run inference
        outputs = self.model.predict(batch)
        
        move_logits = outputs["move_logits"][0]  # (4098,)
        time_logits = outputs.get("time_cls_out", None)  # (256,) if present
        
        # Time prediction
        time_sample_s = 0.0
        mode_time_s = 0.0
        expected_time_s = 0.0
        
        if time_logits is not None:
            time_logits = time_logits[0]  # Remove batch dim
            
            # Calculate time prediction based on settings
            time_temp = float(self.settings.get("time_temperature", 1.0))
            time_top_p = float(self.settings.get("time_top_p", 0.9))
            
            # Calculate softmax for time probs
            exp_logits = np.exp(time_logits - np.max(time_logits))
            time_probs = exp_logits / np.sum(exp_logits)
            
            # Mode (argmax)
            mode_bin = int(np.argmax(time_probs))
            mode_time_s = time_bin_to_seconds(mode_bin, active_clock_s)
            
            # Expected value
            for i in range(len(time_probs)):
                t_sec = time_bin_to_seconds(i, active_clock_s)
                expected_time_s += time_probs[i] * t_sec
            
            # Sample or use mode/expected based on settings
            if self.settings.get("use_mode_time", True):
                time_sample_s = mode_time_s
            elif self.settings.get("use_expected_time", False):
                time_sample_s = expected_time_s
            else:
                # Sample from distribution
                time_bin, _ = sample_time_bin(time_logits, time_temp, time_top_p, self.rng)
                time_sample_s = time_bin_to_seconds(time_bin, active_clock_s)
        
        # Log time prediction if enabled
        if self.settings.get("log_time_history", False):
            self._print(f"info string pred_time sample={time_sample_s:.2f}s mode={mode_time_s:.2f}s expected={expected_time_s:.2f}s")
        
        # Determine temperature based on opening phase
        current_ply = len(self.moves_uci)
        opening_length = int(self.settings.get("opening_length", 10))
        
        if current_ply < opening_length:
            temperature = float(self.settings.get("opening_temperature", 1.2))
        else:
            temperature = float(self.settings["temperature"])
        
        top_p = float(self.settings["top_p"])
        
        if self.settings.get("debug"):
            print(f"# ply={current_ply} temp={temperature:.2f} top_p={top_p:.2f}", file=sys.stderr)
        
        move_idx, prob = sample_move(move_logits, temperature, top_p, self.rng)
        
        # Decode move
        if move_idx >= 4096:
            # Resign/flag - just pick best legal move
            legal = list(canonicalize(self.board).legal_moves)
            if legal:
                # Pick highest prob legal move
                legal_probs = [(mv, move_logits[mv.from_square * 64 + mv.to_square]) for mv in legal]
                legal_probs.sort(key=lambda x: x[1], reverse=True)
                mv = legal_probs[0][0]
            else:
                return None
        else:
            from_sq = move_idx // 64
            to_sq = move_idx % 64
            
            canonical = canonicalize(self.board)
            candidates = [mv for mv in canonical.legal_moves 
                         if mv.from_square == from_sq and mv.to_square == to_sq]
            
            if not candidates:
                # Fallback to best legal move
                legal = list(canonical.legal_moves)
                if legal:
                    legal_probs = [(mv, move_logits[mv.from_square * 64 + mv.to_square]) for mv in legal]
                    legal_probs.sort(key=lambda x: x[1], reverse=True)
                    mv = legal_probs[0][0]
                else:
                    return None
            else:
                # Handle promotions
                promo_moves = [mv for mv in candidates if mv.promotion is not None]
                if promo_moves:
                    # Default to queen promotion
                    mv = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                    if mv not in canonical.legal_moves:
                        mv = promo_moves[0]
                else:
                    mv = candidates[0]
        
        # Convert from canonical (white-to-move) back to real position
        if self.board.turn == chess.BLACK:
            mv = chess.Move(mv.from_square ^ 56, mv.to_square ^ 56, promotion=mv.promotion)
        
        # Update internal clock and time history if InternalClock is enabled
        if self.settings.get("internal_clock", False):
            # Cache this prediction for opponent moves
            self._last_predicted_time_s = time_sample_s
            
            # Update the engine's clock (current side to move)
            if self.board.turn == chess.WHITE:
                self.internal_wtime_s = max(0.0, self.internal_wtime_s - time_sample_s + active_inc_s)
            else:
                self.internal_btime_s = max(0.0, self.internal_btime_s - time_sample_s + active_inc_s)
            
            # Add predicted time to history
            self.pred_time_s_history.append(time_sample_s)
            
            # Debug output
            if self.settings.get("debug_clocks", False):
                self._print(f"info string clocks after_move w={self.internal_wtime_s:.2f}s b={self.internal_btime_s:.2f}s pred_time={time_sample_s:.2f}s history_len={len(self.pred_time_s_history)}")
        
        # Update board state (make the move)
        self.board.push(mv)
        self.moves_uci.append(mv.uci())
        
        return mv
    
    def _handle_stop(self) -> None:
        self._stop_event.set()
    
    def _handle_quit(self) -> None:
        sys.exit(0)
    
    def run(self) -> None:
        """Main UCI loop."""
        while True:
            try:
                line = input()
            except EOFError:
                break
            
            line = line.strip()
            if not line:
                continue
            
            cmd = line.split()[0].lower()
            
            if cmd == "uci":
                self._handle_uci()
            elif cmd == "isready":
                self._handle_isready()
            elif cmd == "ucinewgame":
                self._handle_ucinewgame()
            elif cmd == "position":
                self._handle_position(line)
            elif cmd == "go":
                self._handle_go(line)
            elif cmd == "stop":
                self._handle_stop()
            elif cmd == "quit":
                self._handle_quit()
            elif cmd == "setoption":
                self._handle_setoption(line)
            # Ignore unknown commands


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Marvin ONNX UCI Engine")
    parser.add_argument(
        "--model", "-m",
        type=Path,
        default=None,
        help="Path to ONNX model file (default: inference/marvin_small.onnx)"
    )
    args = parser.parse_args()
    
    try:
        engine = UCIOnnxEngine(model_path=args.model)
        engine.run()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
