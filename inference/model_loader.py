from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import torch


ConfigName = Literal["small", "large", "auto"]


@dataclass(frozen=True)
class LoadedModel:
    model: torch.nn.Module
    config: Dict[str, Any]
    config_name: str
    device: torch.device


def _load_model_module(model_py_path: Path):
    spec = importlib.util.spec_from_file_location("marvin_model", model_py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {model_py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _detect_config_from_state(state: Dict[str, Any]) -> str:
    """Auto-detect which config (small/large) was used based on state_dict weight shapes."""
    # Normalize keys (strip _orig_mod. prefix if present)
    normalized_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    
    # Check d_model size from a known layer to distinguish small vs large
    # small: d_model=448, large: d_model=704
    for key, value in normalized_state.items():
        if "layers.0.attn.q_proj.weight" in key:
            d_model = value.shape[0]
            if d_model >= 700:
                return "large"
            else:
                return "small"
    
    # Default to small if we can't determine
    return "small"


def load_chessformer(
    *,
    model_py_path: str | Path = "model.py",
    config_name: ConfigName = "auto",
    checkpoint_path: str | Path = "inference/marvin_small.pt",
    device: str | torch.device = "cuda",
) -> LoadedModel:
    model_py_path = Path(model_py_path)
    checkpoint_path = Path(checkpoint_path)
    if not model_py_path.exists():
        raise FileNotFoundError(model_py_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    module = _load_model_module(model_py_path)
    device = torch.device(device)

    # Load checkpoint first to potentially auto-detect config
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    
    # Strip _orig_mod. prefix from torch.compile'd checkpoints
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    
    # Auto-detect config if requested
    if config_name == "auto":
        config_name = _detect_config_from_state(state)
        print(f"Auto-detected model config: {config_name}")

    if config_name == "large":
        config = dict(module.CONFIG_LARGE)
    else:  # small (default)
        config = dict(module.CONFIG_SMALL)

    model = module.Chessformer(config).to(device)
    model.load_state_dict(state)

    model.eval()
    return LoadedModel(model=model, config=config, config_name=config_name, device=device)