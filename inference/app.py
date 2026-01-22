import sys
import json
import logging
import queue
import threading
from pathlib import Path
import numpy as np
import torch
import chess
import chess.svg
from flask import Flask, render_template, render_template_string, request, jsonify, Response
import time

from inference.app_settings import DEFAULT_GAME_SETTINGS, DEFAULT_RNG_SEED, INC_S, START_CLOCK_S
from inference.engine_logic import analyze_position as analyze_position_core
from inference.engine_logic import choose_engine_move as choose_engine_move_core
from inference.runtime import (
    count_attn_layers,
    default_device,
    ensure_repo_on_syspath,
    load_default_chessformer,
    resolve_repo_root,
)

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================

# Settings (Defaults)
game_settings = dict(DEFAULT_GAME_SETTINGS)

# Paths
repo_root = resolve_repo_root()
ensure_repo_on_syspath(repo_root)

from inference.encoding import ContextOptions, build_history_from_position, canonicalize, make_model_batch
from inference.chessformer_policy import choose_move
from inference.mcts import MCTSSettings, mcts_choose_move
from inference.sampling import sample_from_logits
from inference.config import get_model_name, print_config

print("Loading model...")
print_config(repo_root)

# Device selection: allow overriding via settings (auto|cuda|cpu)
device_pref = str(game_settings.get("device", "auto")).lower()
if device_pref == "auto":
    device = default_device()
elif device_pref == "cuda":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("[inference] Warning: requested device 'cuda' but CUDA is not available; falling back to 'cpu'.")
        device = torch.device("cpu")
elif device_pref == "cpu":
    device = torch.device("cpu")
else:
    try:
        device = torch.device(device_pref)
    except Exception:
        print(f"[inference] Warning: unrecognized device '{device_pref}'; falling back to auto-detection.")
        device = default_device()

# Model/config selected via inference/config.py or env vars:
#   MARVIN_MODEL=marvin_token_bf16.pt
#   MARVIN_CONFIG=auto
loaded, model, _checkpoint_path = load_default_chessformer(
    repo_root=repo_root, 
    device=device,
)
device = loaded.device
print(f"Model loaded: {get_model_name()} (config: {loaded.config_name})")

# Handle torch.compile wrapper for metadata inspection
orig_model = getattr(model, "_orig_mod", model)
ATTN_NUM_LAYERS = count_attn_layers(model)
ATTN_HAS_SMOLGEN = bool(getattr(orig_model, "smolgen", None) is not None)

# Expose read-only attention metadata to the frontend via the settings blob.
game_settings["attn_num_layers"] = int(ATTN_NUM_LAYERS)
game_settings["attn_has_smolgen"] = bool(ATTN_HAS_SMOLGEN)

# ==========================================
# 2. ANALYSIS LOGIC
# ==========================================
PROMO_INDEX = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}
rng = np.random.default_rng(DEFAULT_RNG_SEED)

def _mirror_square(sq: int) -> int: return sq ^ 56


# Model uses 6 conditioning tokens prepended to 64 square tokens = 70 total tokens
NUM_COND_TOKENS = 6


def _extract_attention_64(
    model: torch.nn.Module,
    *,
    real_turn: chess.Color,
    layer: int,
    head_agg: str,
) -> list[float] | None:
    """Extract a 64x64 attention matrix according to visualization settings.

    Returns flattened list length 4096 in *real* board square indexing (python-chess: a1=0..h8=63).
    Requires the model's attention modules to have `last_attn_probs` populated from a just-completed forward.
    
    The model has 70 tokens (6 conditioning + 64 squares). This extracts only 
    the 64x64 square-to-square attention.

    head_agg:
      - 'avg': average over heads (and over layers if layer == -1)
      - 'max': elementwise max over heads (and over layers if layer == -1)
      - 'smolgen': use smolgen bias only (softmax over bias), averaged over heads
    """

    head_agg = str(head_agg or "avg").lower().strip()
    if head_agg not in ("avg", "max", "smolgen"):
        head_agg = "avg"

    orig_model = getattr(model, "_orig_mod", model)

    def extract_64x64(attn_tensor: torch.Tensor) -> torch.Tensor | None:
        """Extract the 64x64 square-to-square attention from the 70x70 attention tensor."""
        if not torch.is_tensor(attn_tensor) or attn_tensor.ndim != 4:
            return None
        seq_len = attn_tensor.shape[-1]
        if seq_len != 70:
            return None
        # Extract only square-to-square attention (skip first 6 conditioning tokens)
        return attn_tensor[0, :, NUM_COND_TOKENS:, NUM_COND_TOKENS:].float()  # (H, 64, 64)

    # Smolgen-only view (pure dynamic bias).
    if head_agg == "smolgen":
        bias_h: torch.Tensor | None = None
        for mod in orig_model.modules():
            b = getattr(mod, "last_smolgen_bias", None)
            if b is None:
                continue
            bias_h = extract_64x64(b)
            if bias_h is not None:
                break

        if bias_h is None:
            return None

        # Convert smolgen logits to attention probabilities.
        probs_h = torch.softmax(bias_h, dim=-1)  # (H, 64, 64)
        mat = probs_h.mean(dim=0)  # (64, 64)
    else:
        layers_h: list[torch.Tensor] = []
        for mod in orig_model.modules():
            attn = getattr(mod, "last_attn_probs", None)
            if attn is None:
                continue
            attn_64 = extract_64x64(attn)
            if attn_64 is not None:
                layers_h.append(attn_64)  # (H, 64, 64)

        if not layers_h:
            return None

        if int(layer) >= 0:
            idx = int(layer)
            if idx >= len(layers_h):
                idx = len(layers_h) - 1
            layers_h = [layers_h[idx]]

        if head_agg == "avg":
            mats = [a.mean(dim=0) for a in layers_h]  # (64,64) per layer
            mat = torch.stack(mats, dim=0).mean(dim=0)
        else:  # 'max'
            mats = [a.max(dim=0).values for a in layers_h]  # (64,64) per layer
            mat = mats[0]
            for m in mats[1:]:
                mat = torch.maximum(mat, m)

    # Model uses canonical boards (mirrors when black-to-move). Convert back to real-square indexing.
    if real_turn == chess.BLACK:
        perm = torch.tensor([_mirror_square(i) for i in range(64)], device=mat.device, dtype=torch.long)
        mat = mat.index_select(0, perm).index_select(1, perm)

    return mat.detach().cpu().reshape(-1).tolist()
def _softmax_1d(x: torch.Tensor) -> torch.Tensor:
    x = x.float() - torch.max(x)
    return torch.softmax(x, dim=0)
def _time_bin_to_seconds(bin_idx: int, active_clock_s: float) -> float:
    scaled_mid = (bin_idx + 0.5) / 256.0
    # Inverse of sqrt scaling: target = sqrt(time_ratio), so time_ratio = target^2
    return float((scaled_mid ** 2) * max(1e-6, active_clock_s))


def _format_clock_s(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec) // 60
    s = int(sec) % 60
    return f"{m}:{s:02d}"


def _apply_ply_clock(clock_s: float, *, spent_s: float, inc_s: float) -> float:
    # Simple Fischer clock: subtract, clamp at 0, then add increment.
    return max(0.0, float(clock_s) - max(0.0, float(spent_s))) + float(inc_s)


def _clocks_at_cursor(cursor: int) -> dict[chess.Color, float]:
    """Return remaining clocks (seconds) for white/black at the given cursor."""
    start_s = float(game_settings.get("start_clock_s", START_CLOCK_S))
    inc_s = float(game_settings.get("inc_s", INC_S))
    w = start_s
    b = start_s
    hist = game_state.get("history", [])[:cursor]
    for ply_idx, item in enumerate(hist):
        uci = item.get("uci")
        spent = float(item.get("pred_time_s", 0.0))
        mover = chess.WHITE if (ply_idx % 2 == 0) else chess.BLACK
        if uci in ("resign", "flag"):
            # Treat as a ply that consumes time but doesn't change board.
            pass
        if mover == chess.WHITE:
            w = _apply_ply_clock(w, spent_s=spent, inc_s=inc_s)
        else:
            b = _apply_ply_clock(b, spent_s=spent, inc_s=inc_s)
    return {chess.WHITE: w, chess.BLACK: b}

def _canonical_to_real_move(move: chess.Move, real_turn: chess.Color) -> chess.Move:
    if real_turn == chess.WHITE: return move
    return chess.Move(_mirror_square(move.from_square), _mirror_square(move.to_square), promotion=move.promotion)

def _pred_time_history_s_at_cursor(cursor: int) -> list[float]:
    """Return last 8 predicted move times (seconds), newest-first.

    We store one predicted time per played ply in `game_state['history'][i]['pred_time_s']`.
    """
    hist = game_state.get("history", [])[:cursor]
    times = [float(it.get("pred_time_s", 0.0)) for it in hist]
    # Newest-first, pad with zeros to HISTORY_LEN=8.
    out = list(reversed(times[-8:]))
    while len(out) < 8:
        out.append(0.0)
    return out[:8]


def analyze_position(
    board: chess.Board,
    moves_uci: list[str],
    *,
    active_clock_s: float,
    opponent_clock_s: float,
    time_history_s: list[float] | None = None,
) -> dict:
    return analyze_position_core(
        model=model,
        device=loaded.device,
        settings=game_settings,
        rng=rng,
        board=board,
        moves_uci=moves_uci,
        active_clock_s=active_clock_s,
        opponent_clock_s=opponent_clock_s,
        active_inc_s=float(game_settings.get("inc_s", INC_S)),
        opponent_inc_s=float(game_settings.get("inc_s", INC_S)),
        time_history_s=time_history_s,
    )

def format_stats_html(data: dict) -> str:
    if not data: return "<div style='color:#666;padding:10px'><i>Evaluating...</i></div>"
    def mk_bar(p, c='#4caf50'): return f"<div style='background:{c}; width:{max(0,min(100,p*100))}%; height:100%;'></div>"
    
    rows = "".join([f"<div class='row'><span class='lbl'>{m['label']}</span><div class='bar-bg'>{mk_bar(m['prob'], '#2196F3')}</div><span class='val'>{m['prob']:.1%}</span></div>" for m in data['top_moves']])
    
    extras = f"<div class='sub'>Resign: {data['resign']:.1%} | Flag: {data['flag']:.1%}</div>"
    
    policy_html = f"<div class='panel'><h3>Policy (Effective)</h3><div class='policy-scroll'>{rows}</div>{extras}</div>"

    w, d, l = data['wdl']['w'], data['wdl']['d'], data['wdl']['l']
    wdl_html = f"<div class='panel'><h3>WDL</h3><div style='display:flex;height:8px;border-radius:2px;overflow:hidden;'><div style='width:{w*100}%;background:#4caf50'></div><div style='width:{d*100}%;background:#9e9e9e'></div><div style='width:{l*100}%;background:#f44336'></div></div><div class='sub'>W {w:.1%} D {d:.1%} L {l:.1%}</div></div>"
    
    val_html = f"<div class='panel'><h3>Value (Win%)</h3><div class='big-val'>{data['value']:.1%} <span class='err'>±{data['value_error']:.1%}</span></div></div>"
    
    sampled = data.get('time_sample_s', None)
    sampled_prob = data.get('time_sample_prob', None)
    expected_t = data.get('expected_time_s', None)
    mode_t = data.get('mode_time_s', None)
    
    time_dist = data.get('time_dist', [])
    if time_dist:
        max_p = max(d['prob'] for d in time_dist) if time_dist else 1.0
        bars = []
        
        # Find closest bin to mode_t to highlight it
        mode_idx = -1
        if mode_t is not None:
            min_diff = float('inf')
            for i, item in enumerate(time_dist):
                diff = abs(item['sec'] - mode_t)
                if diff < min_diff:
                    min_diff = diff
                    mode_idx = i

        for i, item in enumerate(time_dist):
            h = (item['prob'] / max_p) * 100
            color = '#2196F3'
            if i == mode_idx:
                color = '#ffeb3b' # Yellow for Mode
            bars.append(f"<div title='{item['sec']:.1f}s ({item['prob']:.1%})' style='flex:1; background:{color}; height:{h}%; border-radius:1px 1px 0 0; min-width:2px;'></div>")
        
        start_t = time_dist[0]['sec']
        end_t = time_dist[-1]['sec']
        
        marker_html = ""
        # Show Expected marker (Orange, dashed)
        if expected_t is not None:
            closest_i = -1
            min_diff = float('inf')
            for i, item in enumerate(time_dist):
                diff = abs(item['sec'] - expected_t)
                if diff < min_diff:
                    min_diff = diff
                    closest_i = i
            
            if closest_i >= 0:
                N = len(time_dist)
                pct = ((closest_i + 0.5) / N) * 100
                marker_html += f"<div style='position:absolute; left:{pct}%; bottom:0; height:100%; width:1px; border-left:1px dashed #ff9800; opacity:0.9; pointer-events:none; z-index:9;' title='Expected: {expected_t:.1f}s'></div>"

        graph_html = f"""
        <div style="position:relative; height:60px; margin-top:8px; background:#111; padding:4px; border-radius:4px;">
            <div style="display:flex; align-items:flex-end; height:100%; gap:1px;">
                {"".join(bars)}
            </div>
            {marker_html}
        </div>
        <div style="display:flex; justify-content:space-between; font-size:10px; color:#666; margin-top:2px; padding:0 2px;">
            <span>{start_t:.1f}s</span>
            <span>{end_t:.1f}s</span>
        </div>
        """
        
        sampled_txt = ""
        if sampled is not None:
            sampled_txt = f"<div class='sub'>Sampled: <b style='color:#eee'>{sampled:.1f}s</b> <span style='color:#666'>({sampled_prob:.1%})</span>"
            if mode_t is not None:
                sampled_txt += f" | Mode: <b style='color:#ffeb3b'>{mode_t:.1f}s</b>"
            if expected_t is not None:
                sampled_txt += f" | Exp: <span style='color:#ff9800'>{expected_t:.1f}s</span>"
            sampled_txt += "</div>"
            
        time_top_p = data.get('time_top_p', 0.95)
        time_html = f"<div class='panel'><h3>Time Distribution ({time_top_p:.0%})</h3>{sampled_txt}{graph_html}</div>"
    else:
        time_html = ""

    # Engine Inputs Panel
    # Removed from here, now sent separately as 'engine_inputs'
    return f"""
    <div class="stats-grid">
        <div style="grid-column: span 2">{policy_html}</div>
        {wdl_html}{val_html}<div style="grid-column: span 2">{time_html}</div>
    </div>
    """

def format_mcts_stats_html(stats: dict) -> str:
    root_val = stats.get("root_value", 0.0)
    children = stats.get("children", [])
    
    # Root Value (Static)
    val_html = f"<div class='panel'><h3>MCTS Static Value</h3><div class='big-val'>{root_val:+.2f}</div></div>"
    
    # Children List
    rows = ""
    max_visits = children[0]['visits'] if children else 1
    
    # Header
    rows += """
    <div class='row' style='font-size:10px; color:#888; border-bottom:1px solid #333; padding-bottom:2px; margin-bottom:4px;'>
        <span style='width:40px;'>Move</span>
        <span style='flex:1;'>Visits</span>
        <span style='width:30px; text-align:right;'>N</span>
        <span style='width:40px; text-align:right;'>Q</span>
        <span style='width:30px; text-align:right;'>P</span>
    </div>
    """

    for child in children:
        move = child['move']
        visits = child['visits']
        q = child['q']
        prior = child['prior']
        
        # Bar for visits
        width = (visits / max_visits) * 100
        bar = f"<div style='background:#4caf50; width:{width}%; height:100%;'></div>"
        
        rows += f"""
        <div class='row' style='font-size:11px; gap:4px;'>
            <span class='lbl' style='width:40px; font-family:monospace;'>{move}</span>
            <div class='bar-bg' style='flex:1;'>{bar}</div>
            <span class='val' style='width:30px; text-align:right;'>{visits}</span>
            <span class='val' style='width:40px; text-align:right; color:#aaa;'>{q:+.2f}</span>
            <span class='val' style='width:30px; text-align:right; color:#666;'>{prior:.0%}</span>
        </div>
        """
        
    list_html = f"<div class='panel'><h3>MCTS Search ({len(children)})</h3><div class='policy-scroll' style='max-height:250px;'>{rows}</div></div>"
    
    return f"""
    <div class="stats-grid">
        <div style="grid-column: span 2">{list_html}</div>
        {val_html}
    </div>
    """

# ==========================================
# 3. WEB SERVER & STATE
# ==========================================

app = Flask(__name__)

game_state = {
    "history": [], 
    "cursor": 0,
    # Cache per-position sampled time for the *current* position.
    # Key: cursor index, Value: {fen, time_temperature, time_top_p, time_sample_s, time_sample_prob}
    "pos_cache": {},
    # MCTS tree reuse state
    "last_mcts_result": None,  # MCTSResult from last search
    "last_mcts_ply": -1,  # ply count when last MCTS search was done
}

# MCTS progress streaming
mcts_progress_queue = queue.Queue()
mcts_progress_lock = threading.Lock()
mcts_final_stats = None  # Store final stats for display after MCTS completes


def _mcts_progress_callback(stats: dict) -> None:
    """Called by MCTS during search to report progress."""
    global mcts_final_stats
    try:
        # Non-blocking put - if queue is full, skip this update
        mcts_progress_queue.put_nowait(stats)
        # Also store as final stats (will be overwritten each iteration)
        with mcts_progress_lock:
            mcts_final_stats = stats
    except queue.Full:
        pass


def _clear_pos_cache() -> None:
    game_state["pos_cache"] = {}


def _clear_mcts_tree() -> None:
    """Clear the MCTS tree reuse state."""
    game_state["last_mcts_result"] = None
    game_state["last_mcts_ply"] = -1


def get_board_at_cursor():
    board = chess.Board()
    for i in range(game_state["cursor"]):
        move_uci = game_state["history"][i]['uci']
        board.push(chess.Move.from_uci(move_uci))
    return board

def get_uci_list_at_cursor():
    return [item['uci'] for item in game_state["history"][:game_state["cursor"]]]

def prepare_response(board):
    fen = board.fen()
    moves_uci = get_uci_list_at_cursor()
    cursor = game_state["cursor"]
    time_hist = _pred_time_history_s_at_cursor(cursor)
    clocks = _clocks_at_cursor(cursor)
    active_clock = float(clocks[board.turn])
    current_stats = analyze_position(
        board,
        moves_uci,
        active_clock_s=active_clock,
        opponent_clock_s=float(clocks[not board.turn]),
        time_history_s=time_hist,
    )

    # Persist the sampled time for this position so navigating away/back doesn't resample.
    # - If we're in history, show the sampled time that was *actually used* for the next move.
    # - If we're at the latest position, cache the sampled time for the current position.
    if cursor < len(game_state["history"]):
        hist_item = game_state["history"][cursor]
        if "pred_time_s" in hist_item:
            current_stats["time_sample_s"] = float(hist_item["pred_time_s"])
        if "pred_time_prob" in hist_item:
            current_stats["time_sample_prob"] = float(hist_item["pred_time_prob"])
    else:
        cache = game_state.get("pos_cache", {})
        entry = cache.get(cursor)
        meta = {
            "fen": fen,
            "time_temperature": float(game_settings.get("time_temperature", 1.0)),
            "time_top_p": float(game_settings.get("time_top_p", 0.95)),
        }
        if (
            entry
            and entry.get("fen") == meta["fen"]
            and float(entry.get("time_temperature", -1.0)) == meta["time_temperature"]
            and float(entry.get("time_top_p", -1.0)) == meta["time_top_p"]
        ):
            current_stats["time_sample_s"] = float(entry.get("time_sample_s", current_stats.get("time_sample_s", 0.0)))
            current_stats["time_sample_prob"] = float(entry.get("time_sample_prob", current_stats.get("time_sample_prob", 0.0)))
        else:
            cache[cursor] = {
                **meta,
                "time_sample_s": float(current_stats.get("time_sample_s", 0.0)),
                "time_sample_prob": float(current_stats.get("time_sample_prob", 0.0)),
            }
            game_state["pos_cache"] = cache

    # If we are reviewing history, use the originally sampled time for this move
    if cursor < len(game_state["history"]):
        hist_item = game_state["history"][cursor]
        if 'pred_time_s' in hist_item:
            current_stats['time_sample_s'] = float(hist_item['pred_time_s'])
        if 'pred_time_prob' in hist_item:
            current_stats['time_sample_prob'] = float(hist_item['pred_time_prob'])

    # Attach display clocks in human/engine terms.
    human_color = game_settings["human_color"]
    current_stats['clocks'] = {
        'human': float(clocks[human_color]),
        'engine': float(clocks[not human_color]),
    }
    current_html = format_stats_html(current_stats)
    
    # Extract top move UCI for arrow drawing
    top_move_uci = None
    if current_stats.get('top_moves') and len(current_stats['top_moves']) > 0:
        top_move_uci = current_stats['top_moves'][0].get('uci')
    
    # Construct Engine Inputs HTML
    if board.turn == game_settings["human_color"]:
        act_color = "White" if board.turn == chess.WHITE else "Black"
        act_elo = game_settings["human_elo"]
        opp_elo = game_settings["engine_elo"]
        act_clk = clocks[board.turn]
        opp_clk = clocks[not board.turn]
    else:
        act_color = "White" if board.turn == chess.WHITE else "Black"
        act_elo = game_settings["engine_elo"]
        opp_elo = game_settings["human_elo"]
        act_clk = clocks[board.turn]
        opp_clk = clocks[not board.turn]

    castling_str = board.fen().split(' ')[2]
    ep_str = board.fen().split(' ')[3]
    
    inputs_html = f"""
    <div style="font-size:14px; font-weight:bold; margin-bottom:10px; border-bottom:1px solid #333; padding-bottom:5px; margin-top:20px;">ENGINE INPUTS</div>
    <div style="font-size:12px; color:#aaa; line-height:1.8;">
        <div style="display:flex; justify-content:space-between"><span>Side to Move</span> <b style="color:#eee">{act_color}</b></div>
        <div style="display:flex; justify-content:space-between"><span>Active Clock</span> <b style="color:#eee">{_format_clock_s(act_clk)}</b></div>
        <div style="display:flex; justify-content:space-between"><span>Opponent Clock</span> <b style="color:#eee">{_format_clock_s(opp_clk)}</b></div>
        <div style="display:flex; justify-content:space-between"><span>Active ELO</span> <b style="color:#eee">{act_elo}</b></div>
        <div style="display:flex; justify-content:space-between"><span>Opponent ELO</span> <b style="color:#eee">{opp_elo}</b></div>
        <div style="display:flex; justify-content:space-between"><span>Halfmove Clock</span> <b style="color:#eee">{board.halfmove_clock}</b></div>
        <div style="display:flex; justify-content:space-between"><span>Castling</span> <b style="color:#eee">{castling_str}</b></div>
        <div style="display:flex; justify-content:space-between"><span>En Passant</span> <b style="color:#eee">{ep_str}</b></div>
    </div>
    """
    
    prev_html = "<div style='color:#666;font-style:italic;padding:10px'>Start of game.</div>"
    if game_state["cursor"] > 0:
        prev_item = game_state["history"][game_state["cursor"] - 1]
        prev_html = prev_item.get('prev_stats_html', '')
        if not prev_html:
            prev_html = "<div style='padding:10px'>Human moved.</div>"

    # GENERATE PGN for the full line
    pgn_board = chess.Board()
    full_moves = []
    for item in game_state["history"]:
        uci = item['uci']
        if uci in ('resign', 'flag'):
            continue
        full_moves.append(chess.Move.from_uci(uci))
    
    pgn_str = pgn_board.variation_san(full_moves)

    # Get legal moves as UCI strings for Chessground
    legal_moves = [move.uci() for move in board.legal_moves]
    
    # Get last move for highlighting
    last_move = None
    if game_state["cursor"] > 0:
        last_item = game_state["history"][game_state["cursor"] - 1]
        last_uci = last_item.get('uci', '')
        if last_uci and last_uci not in ('resign', 'flag'):
            last_move = [last_uci[:2], last_uci[2:4]]
    
    # Check if in check
    in_check = board.is_check()

    return jsonify({
        "fen": fen,
        "orientation": "white" if game_settings["human_color"] == chess.WHITE else "black",
        "current_stats": current_html,
        "engine_inputs": inputs_html,
        "prev_engine_stats": prev_html,
        "pgn": pgn_str,
        "cursor": game_state["cursor"],
        "max_cursor": len(game_state["history"]),
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else "",
        "attention64": current_stats.get("attention64"),
        "settings": game_settings,
        "legal_moves": legal_moves,
        "last_move": last_move,
        "in_check": in_check,
        "top_move_uci": top_move_uci
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/state', methods=['GET'])
def get_state():
    board = get_board_at_cursor()
    return prepare_response(board)

@app.route('/settings', methods=['POST'])
def update_settings():
    data = request.json
    
    if 'auto_play' in data:
        game_settings['auto_play'] = bool(data['auto_play'])
    
    game_settings['temperature'] = float(data.get('temperature', game_settings['temperature']))
    game_settings['time_temperature'] = float(data.get('time_temperature', game_settings.get('time_temperature', 1.0)))
    game_settings['top_p'] = float(data.get('top_p', game_settings['top_p']))
    game_settings['time_top_p'] = float(data.get('time_top_p', game_settings.get('time_top_p', 0.95)))
    game_settings['human_elo'] = int(data.get('human_elo', game_settings['human_elo']))
    game_settings['engine_elo'] = int(data.get('engine_elo', game_settings['engine_elo']))
    
    if 'human_color' in data:
        game_settings['human_color'] = chess.WHITE if data['human_color'] == 'white' else chess.BLACK
    
    if 'simulate_thinking_time' in data:
        game_settings['simulate_thinking_time'] = bool(data['simulate_thinking_time'])
        
    if 'use_real_time' in data:
        game_settings['use_real_time'] = bool(data['use_real_time'])

    if 'use_mode_time' in data:
        game_settings['use_mode_time'] = bool(data['use_mode_time'])

    if 'use_expected_time' in data:
        game_settings['use_expected_time'] = bool(data['use_expected_time'])

    if 'start_clock_s' in data:
        game_settings['start_clock_s'] = float(data['start_clock_s'])
    
    if 'inc_s' in data:
        game_settings['inc_s'] = float(data['inc_s'])

    # MCTS
    if 'use_mcts' in data:
        game_settings['use_mcts'] = bool(data['use_mcts'])
    if 'mcts_simulations' in data:
        game_settings['mcts_simulations'] = int(float(data['mcts_simulations']))
    if 'mcts_c_puct' in data:
        game_settings['mcts_c_puct'] = float(data['mcts_c_puct'])
    if 'mcts_max_children' in data:
        game_settings['mcts_max_children'] = int(float(data['mcts_max_children']))
    if 'mcts_root_dirichlet_alpha' in data:
        game_settings['mcts_root_dirichlet_alpha'] = float(data['mcts_root_dirichlet_alpha'])
    if 'mcts_root_exploration_frac' in data:
        game_settings['mcts_root_exploration_frac'] = float(data['mcts_root_exploration_frac'])
    if 'mcts_final_temperature' in data:
        game_settings['mcts_final_temperature'] = float(data['mcts_final_temperature'])
    if 'mcts_max_depth' in data:
        game_settings['mcts_max_depth'] = int(float(data['mcts_max_depth']))
    if 'mcts_adaptive' in data:
        game_settings['mcts_adaptive'] = bool(data['mcts_adaptive'])
    if 'mcts_adaptive_scale' in data:
        game_settings['mcts_adaptive_scale'] = float(data['mcts_adaptive_scale'])
    if 'mcts_tree_reuse' in data:
        game_settings['mcts_tree_reuse'] = bool(data['mcts_tree_reuse'])
        if not game_settings['mcts_tree_reuse']:
            _clear_mcts_tree()  # Clear tree when disabling
    if 'show_mcts_stats' in data:
        game_settings['show_mcts_stats'] = bool(data['show_mcts_stats'])
    if 'show_arrows' in data:
        game_settings['show_arrows'] = bool(data['show_arrows'])
    if 'show_attention' in data:
        game_settings['show_attention'] = bool(data['show_attention'])

    if 'attn_layer' in data:
        try:
            game_settings['attn_layer'] = int(float(data['attn_layer']))
        except Exception:
            game_settings['attn_layer'] = int(game_settings.get('attn_layer', -1))

    if 'attn_head_agg' in data:
        v = str(data['attn_head_agg']).lower().strip()
        if v not in ('avg', 'max', 'smolgen'):
            v = 'avg'
        # If smolgen is unavailable, fall back to avg.
        if v == 'smolgen' and not bool(game_settings.get('attn_has_smolgen', False)):
            v = 'avg'
        game_settings['attn_head_agg'] = v

    if 'attn_focus' in data:
        v = str(data['attn_focus']).lower().strip()
        if v not in ('outbound', 'inbound'):
            v = 'outbound'
        game_settings['attn_focus'] = v

    # Sampling-related settings changes should reset the per-position sampled-time cache.
    if any(k in data for k in ('time_temperature', 'time_top_p', 'use_real_time', 'use_mode_time', 'use_expected_time')):
        _clear_pos_cache()
    
    print(f"DEBUG: Settings Updated: SimulateThinkingTime={game_settings.get('simulate_thinking_time')}, RealTime={game_settings.get('use_real_time')}")

    board = get_board_at_cursor()
    
    # If it's the start of the game and human is Black, engine (White) must move.
    # Frontend will trigger engine move if needed.
        
    return prepare_response(board)

@app.route('/navigate', methods=['POST'])
def navigate():
    data = request.json
    action = data.get('action')
    if action == 'start': game_state["cursor"] = 0
    elif action == 'prev': game_state["cursor"] = max(0, game_state["cursor"] - 1)
    elif action == 'next': game_state["cursor"] = min(len(game_state["history"]), game_state["cursor"] + 1)
    elif action == 'end': game_state["cursor"] = len(game_state["history"])
    return get_state()

def _play_engine_move(board):
    if board.is_game_over():
        return

    moves_uci = get_uci_list_at_cursor()
    cursor_engine = game_state["cursor"]
    time_hist_engine = _pred_time_history_s_at_cursor(cursor_engine)
    clocks_engine = _clocks_at_cursor(cursor_engine)
    
    # Use progress callback if MCTS is enabled and show_mcts_stats is on
    progress_cb = None
    if game_settings.get("use_mcts", False) and game_settings.get("show_mcts_stats", False):
        progress_cb = _mcts_progress_callback
    
    # Prepare tree reuse data
    mcts_reuse_root = None
    mcts_reuse_moves = []
    
    if (
        game_settings.get("mcts_tree_reuse", False)
        and game_state.get("last_mcts_result") is not None
        and game_state.get("last_mcts_ply", -1) >= 0
    ):
        last_result = game_state["last_mcts_result"]
        if last_result.chosen_move is not None:
            # Calculate moves played since last search
            moves_since = cursor_engine - game_state["last_mcts_ply"]
            if 0 < moves_since <= 2:
                mcts_reuse_root = last_result.root
                for i in range(game_state["last_mcts_ply"], cursor_engine):
                    if i < len(moves_uci):
                        mcts_reuse_moves.append(chess.Move.from_uci(moves_uci[i]))
    
    out, engine_stats, mcts_stats, mcts_result = choose_engine_move_core(
        model=model,
        device=loaded.device,
        settings=game_settings,
        rng=rng,
        board=board,
        moves_uci=moves_uci,
        active_clock_s=float(clocks_engine[board.turn]),
        opponent_clock_s=float(clocks_engine[not board.turn]),
        active_inc_s=float(game_settings.get("inc_s", INC_S)),
        opponent_inc_s=float(game_settings.get("inc_s", INC_S)),
        time_history_s=time_hist_engine,
        stop_check=None,
        allow_ponder_sleep=True,
        mcts_progress_callback=progress_cb,
        mcts_reuse_root=mcts_reuse_root,
        mcts_reuse_moves=mcts_reuse_moves,
    )
    
    # Store MCTS result for tree reuse
    if mcts_result is not None:
        game_state["last_mcts_result"] = mcts_result
        game_state["last_mcts_ply"] = cursor_engine

    stats_html = format_stats_html(engine_stats)
    engine_pred_time_s = float(engine_stats.get("time_sample_s", 0.0))
    engine_pred_time_prob = float(engine_stats.get("time_sample_prob", 0.0))
    
    # Cap first-move think time to avoid berserk-related artifacts in training data.
    FIRST_MOVE_TIME_CAP_S = 2.0
    if cursor_engine == 0 and engine_pred_time_s > FIRST_MOVE_TIME_CAP_S:
        engine_pred_time_s = FIRST_MOVE_TIME_CAP_S

    if mcts_stats and game_settings.get("show_mcts_stats", False):
        stats_html = format_mcts_stats_html(mcts_stats)
    
    if out.is_resign:
        chosen_txt = "Resigns"
        chosen_html = f"<div class='chosen-move' style='color:#f44336'>Engine Resigns ({out.policy_prob:.1%})</div>"
        full_prev_html = f"<div style='padding:4px'>{stats_html}</div>{chosen_html}"
        game_state["history"].append({
            'uci': 'resign',
            'pred_time_s': engine_pred_time_s,
            'pred_time_prob': engine_pred_time_prob,
            'prev_stats_html': full_prev_html
        })
    elif out.is_flag:
        chosen_txt = "Flags"
        chosen_html = f"<div class='chosen-move' style='color:#f44336'>Engine Flags ({out.policy_prob:.1%})</div>"
        full_prev_html = f"<div style='padding:4px'>{stats_html}</div>{chosen_html}"
        game_state["history"].append({
            'uci': 'flag',
            'pred_time_s': engine_pred_time_s,
            'pred_time_prob': engine_pred_time_prob,
            'prev_stats_html': full_prev_html
        })
    else:
        chosen_txt = board.san(out.move)
        chosen_html = f"<div class='chosen-move'>Chosen: {chosen_txt} ({out.policy_prob:.1%})</div>"
        full_prev_html = f"<div style='padding:4px'>{stats_html}</div>{chosen_html}"
        
        board.push(out.move)
        
        # Ensure history is truncated to cursor before appending (in case of race conditions or logic bugs)
        game_state["history"] = game_state["history"][:game_state["cursor"]]
        
        game_state["history"].append({
            'uci': out.move.uci(),
            'pred_time_s': engine_pred_time_s,
            'pred_time_prob': engine_pred_time_prob,
            'prev_stats_html': full_prev_html
        })
        game_state["cursor"] += 1

@app.route('/move', methods=['POST'])
def make_move():
    data = request.json
    source = data.get('from')
    target = data.get('to')
    promotion = data.get('promotion', None)
    elapsed_s = data.get('elapsed_s', None)
    
    board = get_board_at_cursor()
    
    try:
        move_uci = f"{source}{target}"
        if promotion: move_uci += promotion
        move = chess.Move.from_uci(move_uci)
        
        if move not in board.legal_moves and not promotion:
             if chess.square_rank(move.to_square) in (0, 7) and board.piece_at(move.from_square).piece_type == chess.PAWN:
                 move = chess.Move.from_uci(move_uci + 'q')

        if move in board.legal_moves:
            game_state["history"] = game_state["history"][:game_state["cursor"]]
            _clear_pos_cache()

            # Predict human move time from current position (top-1 bin).
            moves_uci_before = get_uci_list_at_cursor()
            cursor_before = game_state["cursor"]
            time_hist_before = _pred_time_history_s_at_cursor(cursor_before)
            clocks_before = _clocks_at_cursor(cursor_before)
            human_stats = analyze_position(
                board,
                moves_uci_before,
                active_clock_s=float(clocks_before[board.turn]),
                opponent_clock_s=float(clocks_before[not board.turn]),
                time_history_s=time_hist_before,
            )
            
            if game_settings.get("use_real_time", False) and elapsed_s is not None:
                human_pred_time_s = float(elapsed_s)
                human_pred_time_prob = 1.0
            else:
                human_pred_time_s = float(human_stats.get("time_sample_s", 0.0))
                human_pred_time_prob = float(human_stats.get("time_sample_prob", 0.0))
            
            board.push(move)
            game_state["history"].append({
                'uci': move.uci(),
                'pred_time_s': human_pred_time_s,
                'pred_time_prob': human_pred_time_prob,
                'prev_stats_html': f"<div style='padding:10px; color:#aaa'>Human played <b>{board.pop().uci()}</b> ({human_pred_time_s:.1f}s)</div>"
            })
            board.push(move)
            game_state["cursor"] += 1
        else:
            return jsonify({"error": "Illegal move"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if not board.is_game_over():
        # Frontend will trigger engine move
        pass

    return get_state()

@app.route('/engine_move', methods=['POST'])
def engine_move_endpoint():
    global mcts_final_stats
    # Clear any previous MCTS stats
    with mcts_progress_lock:
        mcts_final_stats = None
    # Drain the queue
    while not mcts_progress_queue.empty():
        try:
            mcts_progress_queue.get_nowait()
        except queue.Empty:
            break
    
    board = get_board_at_cursor()
    # Only play if it is NOT the human's turn (i.e. it is engine's turn)
    if not board.is_game_over() and board.turn != game_settings['human_color']:
        _play_engine_move(board)
    return get_state()


@app.route('/mcts_progress')
def mcts_progress_stream():
    """SSE endpoint for streaming MCTS progress updates."""
    def generate():
        while True:
            try:
                # Wait for new progress with timeout
                stats = mcts_progress_queue.get(timeout=0.1)
                yield f"data: {json.dumps(stats)}\n\n"
            except queue.Empty:
                # Send keepalive
                yield f": keepalive\n\n"
    
    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no'
    })


@app.route('/mcts_stats')
def get_mcts_stats():
    """Get the final MCTS stats after search completes."""
    with mcts_progress_lock:
        if mcts_final_stats:
            return jsonify(mcts_final_stats)
        return jsonify(None)


@app.route('/reset', methods=['POST'])
def reset():
    game_state["history"] = []
    game_state["cursor"] = 0
    _clear_pos_cache()
    _clear_mcts_tree()
    
    # Frontend will trigger engine move if needed
        
    return get_state()

if __name__ == '__main__':
    print("Starting Flask Server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)