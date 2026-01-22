"""
ONNX Export Script for Marvin Chess AI

Exports the Chessformer PyTorch model to ONNX format for use with ONNX Runtime.
Supports CUDA acceleration via CUDAExecutionProvider.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import onnx

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from inference.runtime import load_default_chessformer


class ChessformerONNXWrapper(nn.Module):
    """
    Wrapper that unpacks individual tensor arguments into the batch dict
    expected by Chessformer.forward().
    
    ONNX doesn't support dict inputs, so we take individual named tensors.
    
    Returns 7 outputs including promotion logits for proper promotion handling.
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(
        self,
        board_history: torch.Tensor,   # (B, 8, 64) int64
        time_history: torch.Tensor,    # (B, 8) float32
        rep_flags: torch.Tensor,       # (B, 8) float32
        castling: torch.Tensor,        # (B, 4) float32
        ep_mask: torch.Tensor,         # (B, 64) float32
        scalars: torch.Tensor,         # (B, 8) float32
        tc_cat: torch.Tensor,          # (B,) int64
        legal_mask: torch.Tensor,      # (B, 4098) bool
    ):
        batch = {
            'board_history': board_history,
            'time_history': time_history,
            'rep_flags': rep_flags,
            'castling': castling,
            'ep_mask': ep_mask,
            'scalars': scalars,
            'tc_cat': tc_cat,
            'legal_mask': legal_mask,
        }
        # Forward with return_promo=True returns 7 outputs including promotion logits:
        # move_logits, value_out, value_cls_out, value_error_out, time_cls_out, start_square_logits, promo_logits
        outputs = self.model(batch, return_promo=True)
        return outputs


def create_dummy_inputs(batch_size: int, device: torch.device):
    """Create dummy inputs for ONNX export tracing."""
    return (
        torch.randint(0, 13, (batch_size, 8, 64), dtype=torch.int64, device=device),  # board_history
        torch.randn(batch_size, 8, dtype=torch.float32, device=device),                # time_history
        torch.randint(0, 2, (batch_size, 8), dtype=torch.float32, device=device),      # rep_flags
        torch.randint(0, 2, (batch_size, 4), dtype=torch.float32, device=device),      # castling
        torch.zeros(batch_size, 64, dtype=torch.float32, device=device),               # ep_mask
        torch.randn(batch_size, 8, dtype=torch.float32, device=device),                # scalars
        torch.randint(0, 3, (batch_size,), dtype=torch.int64, device=device),          # tc_cat
        torch.ones(batch_size, 4098, dtype=torch.bool, device=device),                 # legal_mask
    )


def export_to_onnx(
    output_path: Path,
    device: str = "cuda",
    opset_version: int = 14,
    batch_size: int = 1,
):
    """Export Chessformer to ONNX format.
    
    Uses the legacy TorchScript-based ONNX exporter for maximum compatibility.
    The model is exported with a fixed batch size of 1 for simplicity.
    """
    print(f"Loading Chessformer model...")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    loaded, model, checkpoint_path = load_default_chessformer(
        repo_root=REPO_ROOT,
        device=device,
        compile_model=False,  # Don't compile for export
    )
    print(f"  Loaded from: {checkpoint_path}")
    print(f"  Config: {loaded.config_name}")
    print(f"  Device: {device}")
    
    # Wrap for ONNX-compatible interface
    wrapper = ChessformerONNXWrapper(model)
    wrapper.eval()
    
    # Create dummy inputs with batch_size=1 (we'll use fixed batch size for simplicity)
    dummy_inputs = create_dummy_inputs(batch_size, device)
    input_names = [
        "board_history", "time_history", "rep_flags", "castling",
        "ep_mask", "scalars", "tc_cat", "legal_mask"
    ]
    output_names = [
        "move_logits", "value_out", "value_cls_out", 
        "value_error_out", "time_cls_out", "start_square_logits", "promo_logits"
    ]
    
    print(f"\nExporting to ONNX (opset {opset_version}, batch_size={batch_size})...")
    print("  Using legacy TorchScript exporter for compatibility...")
    
    # Use legacy exporter by setting dynamo=False explicitly
    # Also disable dynamic axes to avoid reshape issues
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,  # Use legacy TorchScript exporter
        )
    print(f"  Exported to: {output_path}")
    
    # Validate the exported model
    print("\nValidating ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("  [OK] ONNX model is valid")
    
    # Print model info
    print(f"\nModel info:")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Inputs: {[i.name for i in onnx_model.graph.input]}")
    print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")
    
    return output_path


def validate_onnx_vs_pytorch(onnx_path: Path, device: str = "cuda"):
    """Compare ONNX outputs against PyTorch outputs for numerical validation."""
    import onnxruntime as ort
    import numpy as np
    
    print("\n" + "=" * 60)
    print("Validating ONNX vs PyTorch outputs...")
    print("=" * 60)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load PyTorch model
    loaded, model, _ = load_default_chessformer(
        repo_root=REPO_ROOT,
        device=device,
        compile_model=False,
    )
    model.eval()
    
    # Load ONNX model with CUDA provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    active_provider = session.get_providers()[0]
    print(f"ONNX Runtime provider: {active_provider}")
    
    # Create test inputs - use batch_size=1 to match exported model
    batch_size = 1
    dummy_inputs = create_dummy_inputs(batch_size, device)
    input_names = [
        "board_history", "time_history", "rep_flags", "castling",
        "ep_mask", "scalars", "tc_cat", "legal_mask"
    ]
    
    # PyTorch forward (with return_promo=True to match ONNX wrapper)
    with torch.no_grad():
        batch = {name: inp for name, inp in zip(input_names, dummy_inputs)}
        torch_outputs = model(batch, return_promo=True)
    
    # ONNX forward
    ort_inputs = {
        name: inp.cpu().numpy() for name, inp in zip(input_names, dummy_inputs)
    }
    onnx_outputs = session.run(None, ort_inputs)
    
    # Compare outputs
    output_names = ["move_logits", "value_out", "value_cls_out", 
                    "value_error_out", "time_cls_out", "start_square_logits", "promo_logits"]
    
    all_close = True
    for i, (name, torch_out, onnx_out) in enumerate(zip(output_names, torch_outputs, onnx_outputs)):
        torch_arr = torch_out.cpu().numpy()
        max_diff = np.abs(torch_arr - onnx_out).max()
        mean_diff = np.abs(torch_arr - onnx_out).mean()
        
        # Use larger tolerance for float16/bfloat16 models
        rtol, atol = 1e-3, 1e-4
        is_close = np.allclose(torch_arr, onnx_out, rtol=rtol, atol=atol)
        
        status = "[OK]" if is_close else "[FAIL]"
        print(f"  {status} {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        
        if not is_close:
            all_close = False
    
    if all_close:
        print("\n[OK] All outputs match within tolerance!")
    else:
        print("\n[WARN] Some outputs differ beyond tolerance (may be ok for bf16 models)")
    
    return all_close


def benchmark_inference(onnx_path: Path, device: str = "cuda", num_iters: int = 100):
    """Benchmark ONNX Runtime inference speed."""
    import onnxruntime as ort
    import time
    
    print("\n" + "=" * 60)
    print("Benchmarking ONNX Runtime Inference...")
    print("=" * 60)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load ONNX session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(str(onnx_path), sess_options, providers=providers)
    active_provider = session.get_providers()[0]
    print(f"Provider: {active_provider}")
    
    # Prepare inputs
    batch_size = 1
    dummy_inputs = create_dummy_inputs(batch_size, device)
    input_names = [
        "board_history", "time_history", "rep_flags", "castling",
        "ep_mask", "scalars", "tc_cat", "legal_mask"
    ]
    ort_inputs = {
        name: inp.cpu().numpy() for name, inp in zip(input_names, dummy_inputs)
    }
    
    # Warmup
    for _ in range(10):
        session.run(None, ort_inputs)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        session.run(None, ort_inputs)
    elapsed = time.perf_counter() - start
    
    avg_ms = (elapsed / num_iters) * 1000
    positions_per_sec = num_iters / elapsed
    
    print(f"\nResults ({num_iters} iterations, batch_size={batch_size}):")
    print(f"  Average latency: {avg_ms:.2f} ms")
    print(f"  Throughput: {positions_per_sec:.0f} positions/sec")
    
    return avg_ms, positions_per_sec


def main():
    parser = argparse.ArgumentParser(description="Export Chessformer to ONNX")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=REPO_ROOT / "inference" / "marvin_small.onnx",
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        help="Device for export (cuda or cpu)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14 for compatibility)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate ONNX outputs against PyTorch"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true", 
        help="Run inference benchmark"
    )
    
    args = parser.parse_args()
    
    # Export
    output_path = export_to_onnx(
        output_path=args.output,
        device=args.device,
        opset_version=args.opset,
    )
    
    # Validate
    if args.validate:
        validate_onnx_vs_pytorch(output_path, device=args.device)
    
    # Benchmark
    if args.benchmark:
        benchmark_inference(output_path, device=args.device)
    
    print("\n[OK] Done!")


if __name__ == "__main__":
    main()
