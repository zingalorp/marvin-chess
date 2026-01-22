from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import os
import chess

from inference.app_settings import DEFAULT_GAME_SETTINGS, DEFAULT_RNG_SEED, START_CLOCK_S
from inference.engine_logic import choose_engine_move, analyze_position
from inference.mcts import MCTSResult, _Node
from inference.runtime import load_default_chessformer
from inference.config import get_model_name, print_config, get_model_path, get_config_name


def _bool_from_uci(value: str) -> bool:
    v = str(value).strip().lower()
    return v in ("1", "true", "yes", "on")


def _color_from_uci(value: str) -> chess.Color:
    v = str(value).strip().lower()
    if v in ("black", "b"):
        return chess.BLACK
    return chess.WHITE


def _uci_from_color(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


@dataclass
class _Option:
    name: str
    uci_type: str
    default: Any
    min: float | None = None
    max: float | None = None
    combo: list[str] | None = None


class UciEngine:
    def __init__(self) -> None:
        # Load model without compilation for fast startup.
        # Compilation happens lazily in isready if CompileModel option is true.
        # Model/config selected via inference/config.py or env vars:
        #   MARVIN_MODEL=marvin_token_bf16.pt
        #   MARVIN_CONFIG=auto
        self.loaded, self.model, _ckpt = load_default_chessformer(compile_model=False)
        self._model_compiled = False
        print(f"# Model: {get_model_name()} (config: {self.loaded.config_name})", file=sys.stderr)
        
        # By default use a non-deterministic seed so separate engine processes
        # produce different sampled moves. Set the env var `MARVIN_DETERMINISTIC`
        # to any value to force reproducible seeding via `DEFAULT_RNG_SEED`.
        seed = DEFAULT_RNG_SEED if os.environ.get("MARVIN_DETERMINISTIC") else None
        self.rng = np.random.default_rng(seed)

        # Copy app defaults; we'll expose a subset as UCI options.
        self.settings: dict[str, Any] = dict(DEFAULT_GAME_SETTINGS)

        self.board = chess.Board()
        # UCI GUIs typically re-send `position startpos moves ...` before every `go`.
        # Track the *base* position separately so we don't treat that as a full reset.
        self._base_fen: str = chess.Board().fen()
        self.moves_uci: list[str] = []
        self.pred_time_s_history: list[float] = []  # one per ply; opponent moves default to 0.0

        self._search_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._search_thread: threading.Thread | None = None
        self._last_bestmove: str | None = None

        # Guards against stale `bestmove` output when GUIs send new `position`/`go`
        # while a search is still running.
        self._state_generation: int = 0
        self._active_search_id: int = 0

        # Track last `go` clocks so we can infer real move times for history.
        self._has_last_go: bool = False
        self._last_go_wtime_s: float = float(START_CLOCK_S)
        self._last_go_btime_s: float = float(START_CLOCK_S)
        self._last_go_winc_s: float = 0.0
        self._last_go_binc_s: float = 0.0

        # Last seen UCI clocks (seconds). Some adapters can occasionally send a 0 clock
        # for the non-side-to-move; carry forward the last known value to avoid
        # spuriously telling the model the opponent is flagged.
        self._last_seen_wtime_s: float = float(START_CLOCK_S)
        self._last_seen_btime_s: float = float(START_CLOCK_S)

        self.internal_wtime_s: float = float(START_CLOCK_S)
        self.internal_btime_s: float = float(START_CLOCK_S)

        # Tree reuse state for MCTS
        self._last_mcts_result: MCTSResult | None = None
        self._last_mcts_ply: int = -1  # ply count when last MCTS search was done
        self._ponder_thread: threading.Thread | None = None
        self._ponder_stop_event = threading.Event()

        # Real opponent rating from UCI `opponent` command (lichess-bot sends this)
        self._real_opponent_rating: int | None = None
        self._real_opponent_is_engine: bool = False

        self.options = self._build_options()

    def _maybe_compile_model(self) -> None:
        """Compile model with torch.compile if CompileModel option is enabled."""
        if self._model_compiled:
            return
        if not self.settings.get("compile_model", False):
            return
        
        print("Compiling model with torch.compile...", file=sys.stderr)
        start = time.time()
        try:
            self.model = torch.compile(self.model)
            self._model_compiled = True
            elapsed = time.time() - start
            print(f"Model compilation complete in {elapsed:.1f}s", file=sys.stderr)
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}. Using eager mode.", file=sys.stderr)
            self._model_compiled = True  # Don't retry

    def _build_options(self) -> list[_Option]:
        # These mirror the adjustable knobs in `inference/app.py`, excluding attention/UI-only.
        return [
            # UCI 'spin' is integer-only; Cutechess will treat float defaults as 0 and may not allow editing.
            # Use 'string' for float-like settings (lc0-style behavior).
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
            _Option("UseRealRatings", "check", bool(self.settings.get("use_real_ratings", False))),
            _Option("CompileModel", "check", bool(self.settings.get("compile_model", False))),
            _Option("SimulateThinkingTime", "check", bool(self.settings.get("simulate_thinking_time", False))),
            _Option("InternalClock", "check", bool(self.settings.get("internal_clock", False))),
            _Option("DebugClocks", "check", bool(self.settings.get("debug_clocks", False))),

            # Game time control (set by wrapper before game starts for accurate tc_cat)
            # These are in seconds. If set, they override inference from first `go` clocks.
            _Option("GameBaseTime", "string", str(self.settings.get("game_base_time_s", 0))),
            _Option("GameIncrement", "string", str(self.settings.get("game_increment_s", 0))),

            # Controls whether the engine prints resign/flag probabilities each search.
            _Option("LogResignProbs", "check", bool(self.settings.get("log_resign_probs", False))),
            _Option("LogTimeHistory", "check", bool(self.settings.get("log_time_history", False))),
            _Option("LogMctsStats", "check", bool(self.settings.get("log_mcts_stats", False))),

            # Human-like terminal behaviors (UCI has no standard resign/flag moves).
            # We expose these so wrappers can react to `info string action=...`.
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
            _Option("Ponder", "check", bool(self.settings.get("ponder", False))),
        ]

    def _print(self, line: str) -> None:
        sys.stdout.write(line.rstrip("\n") + "\n")
        sys.stdout.flush()

    def _time_history_last8_newest_first(self) -> list[float]:
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
        # In UCI mode, the engine is always the side-to-move.
        self.settings["human_color"] = (not self.board.turn)
        # Clear tree reuse state
        self._last_mcts_result = None
        self._last_mcts_ply = -1
        # Clear opponent info (will be re-sent by lichess-bot each game)
        self._real_opponent_rating = None
        self._real_opponent_is_engine = False

    def _invalidate_search_locked(self) -> threading.Thread | None:
        """Invalidate any in-flight search so it can't print/update stale output."""

        self._active_search_id += 1
        self._state_generation += 1
        t = self._search_thread
        if t is not None and t.is_alive():
            self._stop_event.set()
        return t

    def _set_position(self, *, board: chess.Board, moves: list[str]) -> None:
        # Determine the base position (startpos or provided FEN).
        base_fen = board.fen() if board.fen() != chess.Board().fen() else chess.Board().fen()
        base_board = chess.Board(fen=base_fen)

        # If base changed (different FEN), hard reset.
        if base_fen != self._base_fen:
            self._base_fen = base_fen
            self.board = base_board.copy(stack=False)
            self.moves_uci = []
            self.pred_time_s_history = []

        # Find common prefix with our current move list.
        common = 0
        max_common = min(len(self.moves_uci), len(moves))
        while common < max_common and self.moves_uci[common] == moves[common]:
            common += 1

        # If GUI rewound/changed moves, rebuild deterministically from base, preserving time history for the common prefix.
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
            # Append only newly-seen moves, keeping our existing time history intact.
            for uci in moves[common:]:
                mv = chess.Move.from_uci(uci)
                if mv not in self.board.legal_moves:
                    raise ValueError(f"Illegal move in position: {uci}")
                
                if self.settings.get("internal_clock", False):
                    # Estimate time for the move about to be made (opponent's move)
                    # We use the current internal clocks.
                    active_clock = self.internal_wtime_s if self.board.turn == chess.WHITE else self.internal_btime_s
                    opp_clock = self.internal_btime_s if self.board.turn == chess.WHITE else self.internal_wtime_s
                    active_inc = self._last_go_winc_s if self.board.turn == chess.WHITE else self._last_go_binc_s
                    opp_inc = self._last_go_binc_s if self.board.turn == chess.WHITE else self._last_go_winc_s
                    
                    time_hist = self._time_history_last8_newest_first()
                    
                    est_stats = analyze_position(
                        model=self.model,
                        device=self.loaded.device,
                        settings=self.settings,
                        rng=self.rng,
                        board=self.board,
                        moves_uci=self.moves_uci,
                        active_clock_s=active_clock,
                        opponent_clock_s=opp_clock,
                        active_inc_s=active_inc,
                        opponent_inc_s=opp_inc,
                        time_history_s=time_hist,
                    )
                    est_t = float(est_stats.get("time_sample_s", 0.0))
                    
                    # Update opponent's internal clock
                    if self.board.turn == chess.WHITE:
                        self.internal_wtime_s = max(0.0, self.internal_wtime_s - est_t + active_inc)
                    else:
                        self.internal_btime_s = max(0.0, self.internal_btime_s - est_t + active_inc)
                    
                    self.pred_time_s_history.append(est_t)
                else:
                    self.pred_time_s_history.append(0.0)

                self.board.push(mv)
                self.moves_uci.append(uci)

        # In UCI mode, the engine is always the side-to-move.
        self.settings["human_color"] = (not self.board.turn)

    def _handle_uci(self) -> None:
        self._print("id name marvin-chessformer")
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
                # Default must be a single token; avoid spaces.
                self._print(f"option name {opt.name} type string default {opt.default}")
            else:  # spin
                mn = int(opt.min) if opt.min is not None else 0
                mx = int(opt.max) if opt.max is not None else 100
                self._print(f"option name {opt.name} type spin default {opt.default} min {mn} max {mx}")

        self._print("uciok")

    def _handle_opponent(self, line: str) -> None:
        """Handle UCI 'opponent' command sent by lichess-bot.
        
        Format: opponent [title <title>] [name <name>] [rating <rating>] [type <computer|human>]
        
        We extract the rating and type to use for the model's opponent_elo input
        when UseRealRatings is enabled.
        """
        tokens = line.strip().split()
        
        rating: int | None = None
        is_engine: bool = False
        
        i = 1  # skip 'opponent' token
        while i < len(tokens):
            if tokens[i] == "rating" and i + 1 < len(tokens):
                try:
                    rating = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2
            elif tokens[i] == "type" and i + 1 < len(tokens):
                is_engine = tokens[i + 1].lower() == "computer"
                i += 2
            elif tokens[i] in ("title", "name") and i + 1 < len(tokens):
                i += 2  # skip title/name and its value
            else:
                i += 1
        
        self._real_opponent_rating = rating
        self._real_opponent_is_engine = is_engine
        
        if rating is not None:
            print(f"# Opponent rating: {rating} (is_engine={is_engine})", file=sys.stderr)

    def _handle_setoption(self, line: str) -> None:
        # setoption name <name> [value <value>]
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
            name = " ".join(tokens[name_idx + 1 :])
            value = ""
        else:
            name = " ".join(tokens[name_idx + 1 : value_idx])
            value = " ".join(tokens[value_idx + 1 :])

        name_key = name.strip().lower()

        def set_setting(key: str, v: Any) -> None:
            self.settings[key] = v

        if name_key == "temperature":
            set_setting("temperature", float(value))
        elif name_key == "topp":
            set_setting("top_p", float(value))
        elif name_key == "timetemperature":
            set_setting("time_temperature", float(value))
        elif name_key == "timetopp":
            set_setting("time_top_p", float(value))
        elif name_key == "openingtemperature":
            set_setting("opening_temperature", float(value))
        elif name_key == "openinglength":
            set_setting("opening_length", int(float(value)))
        elif name_key == "usemodetime":
            set_setting("use_mode_time", _bool_from_uci(value))
        elif name_key == "useexpectedtime":
            set_setting("use_expected_time", _bool_from_uci(value))
        elif name_key == "userealtime":
            set_setting("use_real_time", _bool_from_uci(value))
        elif name_key == "humanelo":
            set_setting("human_elo", int(float(value)))
        elif name_key == "engineelo":
            set_setting("engine_elo", int(float(value)))
        elif name_key == "userealratings":
            set_setting("use_real_ratings", _bool_from_uci(value))
        elif name_key == "compilemodel":
            set_setting("compile_model", _bool_from_uci(value))
        elif name_key == "simulatethinkingtime":
            set_setting("simulate_thinking_time", _bool_from_uci(value))
        elif name_key == "internalclock":
            set_setting("internal_clock", _bool_from_uci(value))
        elif name_key == "debugclocks":
            set_setting("debug_clocks", _bool_from_uci(value))
        elif name_key == "gamebasetime":
            # Game base time in seconds (initial clock, not remaining)
            base_s = float(value)
            set_setting("game_base_time_s", base_s)
            # Also set start_clock_s so tc_cat is computed correctly
            if base_s > 0:
                set_setting("start_clock_s", base_s)
        elif name_key == "gameincrement":
            # Game increment in seconds
            set_setting("game_increment_s", float(value))
        elif name_key == "logresignprobs":
            set_setting("log_resign_probs", _bool_from_uci(value))
        elif name_key == "logtimehistory":
            set_setting("log_time_history", _bool_from_uci(value))
        elif name_key == "logmctsstats":
            set_setting("log_mcts_stats", _bool_from_uci(value))
        elif name_key == "enableresign":
            set_setting("enable_resign", _bool_from_uci(value))
        elif name_key == "resignthreshold":
            set_setting("resign_threshold", float(value))
        elif name_key == "minresignply":
            # Minimum ply (half-moves) before the engine will announce resign.
            set_setting("resign_min_ply", int(float(value)))
        elif name_key == "enableflag":
            set_setting("enable_flag", _bool_from_uci(value))
        elif name_key == "flagthreshold":
            set_setting("flag_threshold", float(value))
        elif name_key == "usemcts":
            set_setting("use_mcts", _bool_from_uci(value))
        elif name_key == "mctssimulations":
            set_setting("mcts_simulations", int(float(value)))
        elif name_key == "mctscpuct":
            set_setting("mcts_c_puct", float(value))
        elif name_key == "mctsmaxchildren":
            set_setting("mcts_max_children", int(float(value)))
        elif name_key == "mctsrootdirichletalpha":
            set_setting("mcts_root_dirichlet_alpha", float(value))
        elif name_key == "mctsrootexplorationfrac":
            set_setting("mcts_root_exploration_frac", float(value))
        elif name_key == "mctsfinaltemperature":
            set_setting("mcts_final_temperature", float(value))
        elif name_key == "mctsfinaltop_p" or name_key == "mctsfinaltopp":
            set_setting("mcts_final_top_p", float(value))
        elif name_key == "mctsmaxdepth":
            set_setting("mcts_max_depth", int(float(value)))
        elif name_key == "mctsadaptive":
            set_setting("mcts_adaptive", _bool_from_uci(value))
        elif name_key == "mctsadaptivescale":
            set_setting("mcts_adaptive_scale", float(value))
        elif name_key == "mctsfpu":
            set_setting("mcts_fpu_reduction", float(value))
        elif name_key == "mctscontempt":
            set_setting("mcts_contempt", float(value))
        elif name_key == "mctssimulatetime":
            set_setting("mcts_simulate_time", _bool_from_uci(value))
        elif name_key == "mctsstartply":
            set_setting("mcts_start_ply", int(float(value)))
        elif name_key == "mctstreereuse":
            set_setting("mcts_tree_reuse", _bool_from_uci(value))
        elif name_key == "ponder":
            set_setting("ponder", _bool_from_uci(value))

        # Echo back what option was set so wrappers and logs can verify it took effect.
        try:
            self._print(f"info string setoption {name}={value}")
        except Exception:
            pass

    def _parse_position(self, line: str) -> tuple[chess.Board, list[str]]:
        tokens = line.strip().split()
        if len(tokens) < 2:
            return chess.Board(), []

        idx = 1
        if tokens[idx] == "startpos":
            board = chess.Board()
            idx += 1
        elif tokens[idx] == "fen":
            idx += 1
            # FEN is 6 tokens; consume until 'moves' or end.
            fen_parts: list[str] = []
            while idx < len(tokens) and tokens[idx] != "moves":
                fen_parts.append(tokens[idx])
                idx += 1
            fen = " ".join(fen_parts)
            board = chess.Board(fen=fen)
        else:
            board = chess.Board()

        moves: list[str] = []
        if idx < len(tokens) and tokens[idx] == "moves":
            moves = tokens[idx + 1 :]

        return board, moves

    def _maybe_update_real_time_history(
        self,
        *,
        wtime_s: float,
        btime_s: float,
        winc_s: float,
        binc_s: float,
    ) -> None:
        """Update last two ply move-times from clock deltas.

        UCI gives remaining clocks *before* the engine's move.
        Between consecutive `go` calls (engine-only), two plies happen:
          1) engine move, then increment applied to engine clock
          2) opponent move, then increment applied to opponent clock

        With Fischer clocks: time_after = time_before - spent + inc => spent = time_before + inc - time_after
        """

        if not bool(self.settings.get("use_real_time", False)):
            return
        if not self._has_last_go:
            return

        prev_w = float(self._last_go_wtime_s)
        prev_b = float(self._last_go_btime_s)

        # Use current increments (normally constant for the game).
        inc_w = float(winc_s)
        inc_b = float(binc_s)

        curr_w = float(wtime_s)
        curr_b = float(btime_s)

        spent_w = max(0.0, prev_w + inc_w - curr_w)
        spent_b = max(0.0, prev_b + inc_b - curr_b)

        if bool(self.settings.get("debug_clocks", False)):
            stm = "W" if self.board.turn == chess.WHITE else "B"
            self._print(
                "info string real_time_delta "
                f"stm={stm} prev_w={prev_w:.3f} prev_b={prev_b:.3f} "
                f"curr_w={curr_w:.3f} curr_b={curr_b:.3f} "
                f"winc={inc_w:.3f} binc={inc_b:.3f} "
                f"spent_w={spent_w:.3f} spent_b={spent_b:.3f}"
            )

        # At a `go` call, it's engine-to-move, so the last ply in the moves list is opponent's move.
        # Only update the OPPONENT's time ([-1]) with real clock data, not the engine's ([-2]).
        # The engine's predicted time should remain as predicted by the model, not overwritten
        # with actual response time (which may be instant if SimulateThinkingTime=false).
        # This prevents a feedback loop where fast engine response → model sees fast history → predicts fast.
        if len(self.pred_time_s_history) >= 1:
            last_mover = not self.board.turn  # opponent just moved
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
                
                # Prepare tree reuse data
                mcts_reuse_root: _Node | None = None
                mcts_reuse_moves: list[chess.Move] = []
                
                if (
                    bool(self.settings.get("mcts_tree_reuse", False))
                    and self._last_mcts_result is not None
                    and self._last_mcts_ply >= 0
                    and self._last_mcts_result.chosen_move is not None
                ):
                    # Calculate moves played since last search
                    moves_since = current_ply - self._last_mcts_ply
                    if 0 < moves_since <= 2:  # Expect 2 moves: our move + opponent's move
                        # Get the moves that were played since last search
                        mcts_reuse_root = self._last_mcts_result.root
                        for i in range(self._last_mcts_ply, current_ply):
                            if i < len(moves_uci):
                                mcts_reuse_moves.append(chess.Move.from_uci(moves_uci[i]))

            if bool(self.settings.get("log_time_history", False)):
                # Print newest-first history so it matches model input ordering.
                self._print(f"info string time_history {time_hist}")

            # Determine active/opponent clocks based on side-to-move.
            # The "active" clock is ALWAYS the clock for the side whose turn it is.
            # UCI provides wtime (White's remaining time) and btime (Black's remaining time).
            # When board.turn == WHITE, we are White, so our clock is wtime.
            # When board.turn == BLACK, we are Black, so our clock is btime.
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

            if bool(self.settings.get("log_time_history", False)):
                # Explicit debug: show exactly what clock was assigned to whom
                engine_color = "WHITE" if board.turn == chess.WHITE else "BLACK"
                self._print(
                    f"info string clock_assignment engine_plays={engine_color} "
                    f"wtime={wtime_s:.1f}s btime={btime_s:.1f}s => "
                    f"active_clock={active_clock_s:.1f}s opp_clock={opponent_clock_s:.1f}s"
                )

            # If UseRealRatings is enabled and we have a real opponent rating,
            # temporarily override human_elo for this move computation
            use_real_ratings = bool(self.settings.get("use_real_ratings", False))
            original_human_elo = self.settings["human_elo"]
            if use_real_ratings and self._real_opponent_rating is not None:
                self.settings["human_elo"] = self._real_opponent_rating
                self._print(f"info string using real opponent rating: {self._real_opponent_rating}")

            out, engine_stats, _mcts_stats, mcts_result = choose_engine_move(
                model=self.model,
                device=self.loaded.device,
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

            # Restore original human_elo if we overrode it
            if use_real_ratings and self._real_opponent_rating is not None:
                self.settings["human_elo"] = original_human_elo

            # Optional: surface resign/flag head signals to wrappers.
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

            # Respect a minimum ply threshold to avoid premature resignations
            min_resign_ply = int(self.settings.get("resign_min_ply", 20))
            current_ply = len(moves_uci)

            if enable_resign and resign_p >= resign_thr and current_ply >= min_resign_ply:
                self._print(f"info string action=resign resign_p={resign_p:.4f} flag_p={flag_p:.4f}")
            if enable_flag and flag_p >= flag_thr and current_ply >= min_resign_ply:
                self._print(f"info string action=flag resign_p={resign_p:.4f} flag_p={flag_p:.4f}")
            # Optional logging of probabilities even when thresholds are not met.
            if bool(self.settings.get("log_resign_probs", False)):
                self._print(f"info string probs resign_p={resign_p:.4f} flag_p={flag_p:.4f}")

            # Optional logging of lightweight MCTS diagnostics.
            try:
                if bool(self.settings.get("log_mcts_stats", False)) and isinstance(_mcts_stats, dict):
                    mn = int(_mcts_stats.get("mcts_nodes", 0))
                    mps = float(_mcts_stats.get("mcts_nodes_per_s", 0.0))
                    tree_reused = mcts_reuse_root is not None and len(mcts_reuse_moves) > 0
                    self._print(f"info string mcts_nodes={mn} mps={mps:.1f} tree_reused={tree_reused}")
            except Exception:
                pass

            # Log predicted time for debugging
            if bool(self.settings.get("log_time_history", False)):
                pred_t = float(engine_stats.get("time_sample_s", 0.0)) if isinstance(engine_stats, dict) else 0.0
                mode_t = float(engine_stats.get("mode_time_s", 0.0)) if isinstance(engine_stats, dict) else 0.0
                expected_t = float(engine_stats.get("expected_time_s", 0.0)) if isinstance(engine_stats, dict) else 0.0
                self._print(f"info string pred_time sample={pred_t:.2f}s mode={mode_t:.2f}s expected={expected_t:.2f}s")

            # UCI GUIs generally require `bestmove` to be a legal move string.
            # The web app supports pseudo-moves (resign/flag) via `move=None`, but UCI does not.
            # Also, we keep playing in claimable-draw positions, so `move=None` should be rare.
            if out.move is not None:
                bestmove = out.move.uci()
            else:
                # If there are no legal moves (mate/stalemate), `0000` is acceptable.
                if not any(board.legal_moves):
                    bestmove = "0000"
                else:
                    # Prefer the highest-prob policy move (already real-uci) when available.
                    fallback_uci: str | None = None
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
                # If a new `position`/`go` superseded this search, discard.
                if search_id != self._active_search_id or state_generation != self._state_generation:
                    return

                self._last_bestmove = bestmove
                
                # Store MCTS result for tree reuse
                if mcts_result is not None:
                    self._last_mcts_result = mcts_result
                    self._last_mcts_ply = len(moves_uci)

                # If our state hasn't changed mid-search, advance it.
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
                    # Store predicted time for this ply (engine move).
                    self.pred_time_s_history.append(pred_t)

            self._print(f"bestmove {bestmove}")
        finally:
            with self._search_lock:
                self._search_thread = None
                self._stop_event.clear()

    def _handle_go(self, line: str) -> None:
        tokens = line.strip().split()

        # Proper UCI parsing: some tokens are flags (ponder/infinite), others are key-value.
        args: dict[str, str] = {}
        flags: set[str] = set()
        i = 1
        while i < len(tokens):
            t = tokens[i]
            if t in ("ponder", "infinite"):
                flags.add(t)
                i += 1
                continue
            if t in ("wtime", "btime", "winc", "binc", "movestogo", "depth", "nodes", "mate", "movetime"):
                if i + 1 < len(tokens):
                    args[t] = tokens[i + 1]
                    i += 2
                    continue
            # Unknown token: skip.
            i += 1

        # Determine the default clock to use when wtime/btime not provided.
        # Prefer game_base_time_s (set via GameBaseTime option before game starts),
        # then fall back to start_clock_s (inferred from first go), then START_CLOCK_S.
        default_clock_s = float(self.settings.get("game_base_time_s", 0))
        if default_clock_s <= 0:
            default_clock_s = float(self.settings.get("start_clock_s", START_CLOCK_S))
        default_clock_ms = int(default_clock_s * 1000)

        # Clocks are milliseconds in UCI.
        wtime_ms = float(args.get("wtime", str(default_clock_ms)))
        btime_ms = float(args.get("btime", str(default_clock_ms)))
        wtime_s = wtime_ms / 1000.0
        btime_s = btime_ms / 1000.0

        # Default increment to use when winc/binc not provided.
        default_inc_s = float(self.settings.get("game_increment_s", 0))
        default_inc_ms = int(default_inc_s * 1000)

        # Increments (milliseconds). Used for model context if plumbed.
        winc_s = float(args.get("winc", str(default_inc_ms))) / 1000.0
        binc_s = float(args.get("binc", str(default_inc_ms))) / 1000.0

        # Capture the game's base time for tc_cat calculation.
        # Prefer the explicit GameBaseTime option (set by wrapper before game starts).
        # Fall back to inferring from the first `go` clocks if not set.
        if not self._has_last_go:
            explicit_base = float(self.settings.get("game_base_time_s", 0))
            if explicit_base > 0:
                # Already set via setoption GameBaseTime; don't override
                pass
            else:
                # Infer from first go command (less accurate but better than nothing)
                self.settings["start_clock_s"] = float(max(wtime_s, btime_s))

        if bool(self.settings.get("debug_clocks", False)):
            stm = "W" if self.board.turn == chess.WHITE else "B"
            active_clock_s = float(wtime_s if self.board.turn == chess.WHITE else btime_s)
            opponent_clock_s = float(btime_s if self.board.turn == chess.WHITE else wtime_s)
            game_base = float(self.settings.get("game_base_time_s", 0))
            game_inc = float(self.settings.get("game_increment_s", 0))
            self._print(
                "info string uci_clocks "
                f"stm={stm} wtime={wtime_s:.3f} btime={btime_s:.3f} "
                f"active={active_clock_s:.3f} opp={opponent_clock_s:.3f} "
                f"winc={winc_s:.3f} binc={binc_s:.3f} "
                f"use_real_time={bool(self.settings.get('use_real_time', False))} "
                f"internal_clock={bool(self.settings.get('internal_clock', False))} "
                f"start_clock_s={float(self.settings.get('start_clock_s', START_CLOCK_S)):.3f} "
                f"game_base={game_base:.1f} game_inc={game_inc:.1f}"
            )

        # Robustness: if an adapter ever sends a missing/zero clock for one side,
        # reuse the last seen clock instead of feeding 0.0 into the model.
        # Only do this after we've seen at least one valid `go`.
        if self._has_last_go:
            if wtime_s <= 0.0 and self._last_seen_wtime_s > 0.0:
                wtime_s = float(self._last_seen_wtime_s)
            if btime_s <= 0.0 and self._last_seen_btime_s > 0.0:
                btime_s = float(self._last_seen_btime_s)

        self._last_seen_wtime_s = float(wtime_s)
        self._last_seen_btime_s = float(btime_s)

        if self.settings.get("internal_clock", False):
            # If this is the first go, sync internal clocks to UCI clocks once.
            if not self._has_last_go:
                self.internal_wtime_s = wtime_s
                self.internal_btime_s = btime_s
            
            wtime_s = self.internal_wtime_s
            btime_s = self.internal_btime_s

        # Update time-history from real clock deltas (if enabled).
        with self._search_lock:
            self._maybe_update_real_time_history(wtime_s=wtime_s, btime_s=btime_s, winc_s=winc_s, binc_s=binc_s)

            # Record current clocks for next `go` delta.
            self._has_last_go = True
            self._last_go_wtime_s = float(wtime_s)
            self._last_go_btime_s = float(btime_s)
            self._last_go_winc_s = float(winc_s)
            self._last_go_binc_s = float(binc_s)

        # If a previous search is still running, cancel it so we can respond to the latest `go`.
        prev_t: threading.Thread | None = None
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

        with self._search_lock:
            if self._last_bestmove is not None:
                # If the worker already printed bestmove, printing again is usually harmless but noisy.
                # Avoid duplicates.
                pass

    def loop(self) -> None:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue

            if line == "uci":
                self._handle_uci()
            elif line == "isready":
                self._maybe_compile_model()
                self._print("readyok")
            elif line.startswith("setoption"):
                self._handle_setoption(line)
            elif line == "ucinewgame":
                t: threading.Thread | None = None
                with self._search_lock:
                    t = self._invalidate_search_locked()
                    self._reset_position()
                if t is not None:
                    t.join(timeout=5.0)
            elif line.startswith("position"):
                board, moves = self._parse_position(line)
                t2: threading.Thread | None = None
                with self._search_lock:
                    t2 = self._invalidate_search_locked()
                if t2 is not None:
                    t2.join(timeout=5.0)
                with self._search_lock:
                    self._set_position(board=board, moves=moves)
            elif line.startswith("go"):
                self._handle_go(line)
            elif line.startswith("opponent"):
                self._handle_opponent(line)
            elif line == "stop":
                self._handle_stop()
            elif line == "quit":
                self._handle_stop()
                return


def main() -> None:
    engine = UciEngine()
    engine.loop()


if __name__ == "__main__":
    main()
