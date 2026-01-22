from __future__ import annotations

import sys
from pathlib import Path

import torch

from inference.model_loader import LoadedModel, load_chessformer
from inference.config import get_model_path, get_config_name


def resolve_repo_root(*, cwd: Path | None = None) -> Path:
    cwd = (cwd or Path.cwd()).resolve()
    if (cwd / "model.py").exists():
        return cwd
    if (cwd.parent / "model.py").exists():
        return cwd.parent
    return cwd


def ensure_repo_on_syspath(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_default_chessformer(*, repo_root: Path | None = None, device: torch.device | None = None, compile_model: bool = True, checkpoint_path: Path | None = None, config_name: str | None = None) -> tuple[LoadedModel, torch.nn.Module, Path]:
    repo_root = (repo_root or resolve_repo_root()).resolve()
    ensure_repo_on_syspath(repo_root)

    device = device or default_device()

    model_py_path = repo_root / "model.py"
    # Use provided checkpoint or get from centralized config
    if checkpoint_path is None:
        checkpoint_path = get_model_path(repo_root)
    # Use provided config or get from centralized config
    if config_name is None:
        config_name = get_config_name()
    if not checkpoint_path.exists():
        print(f"Error: model weights are missing. Expected checkpoint at {checkpoint_path}.")
        raise FileNotFoundError(checkpoint_path)

    loaded = load_chessformer(
        model_py_path=model_py_path,
        config_name=config_name,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    model = loaded.model
    model.eval()

    if compile_model:
        try:
            model = torch.compile(model, mode="default")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}. Falling back to eager mode.")

    return loaded, model, checkpoint_path


def count_attn_layers(model: torch.nn.Module) -> int:
    n = 0
    # Handle torch.compile wrapper
    orig_model = getattr(model, "_orig_mod", model)
    for mod in orig_model.modules():
        if hasattr(mod, "last_attn_probs"):
            n += 1
    return n