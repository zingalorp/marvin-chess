"""Trainer for Chessformer using per-token input encoding and legal move masking."""

from __future__ import annotations

import argparse
import copy
import math
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter

from dataset import create_dataloader, count_samples

# Import model
from model import (
    Chessformer,
    CONFIG_LARGE,
    CONFIG_SMALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chessformer trainer")
    parser.add_argument("--data-dir", default="data", help="Root directory containing train/val/test splits")
    parser.add_argument("--config", choices=["large", "small"], default="small")
    parser.add_argument("--batch-size", type=int, default=640)  # 640 for small config, 256 for large config
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.25e-4)  # 1.25e-4 for small config, 5e-5 for large config
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)  # Training on noisy multi modal data so smaller batch sizes are fine? idk
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=50000, help="Run validation every N optimizer steps")
    parser.add_argument("--val-batches", type=int, default=5000, help="Max batches per validation run (0 for full)")
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--prefetch-factor", type=int, default=4)  # Lower but more workers
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="max-autotune", 
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (default: max-autotune)")
    parser.add_argument("--gradient-checkpointing", action="store_true",  # Training on noisy data doesn't benefit as much from large batch sizes? So keep this off.
                        help="Enable gradient checkpointing to reduce memory usage")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Number of warmup steps for LR scheduler")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-wait", type=int, default=2)
    parser.add_argument("--profile-warmup", type=int, default=2)
    parser.add_argument("--profile-active", type=int, default=6)
    parser.add_argument("--profile-repeat", type=int, default=1)
    parser.add_argument("--profile-dir", default=None)
    parser.add_argument("--profile-stack", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Loss weights
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--value-cls-weight", type=float, default=0.3)
    parser.add_argument("--value-error-weight", type=float, default=0.1)
    parser.add_argument("--time-weight", type=float, default=0.5)
    parser.add_argument("--start-square-weight", type=float, default=0.1)
    parser.add_argument("--promo-weight", type=float, default=0.1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_precision(device: torch.device, disable_tf32: bool) -> None:
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = not disable_tf32
        torch.backends.cudnn.allow_tf32 = not disable_tf32
        # Enable cudnn benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True


def create_profiler(args, run_dir: Path, device: torch.device, available_steps: Optional[int] = None):
    if not args.profile:
        return nullcontext()

    wait = args.profile_wait
    warmup = args.profile_warmup
    active = args.profile_active

    if available_steps is not None:
        cycle = wait + warmup + active
        if cycle > available_steps:
            wait = min(wait, max(0, available_steps - 2))
            warmup = min(warmup, max(0, available_steps - wait - 1))
            active = max(1, available_steps - wait - warmup)
            print(f"[profiler] Adjusted to wait={wait}, warmup={warmup}, active={active}")

    trace_dir = Path(args.profile_dir) if args.profile_dir else run_dir / "profiler"
    trace_dir.mkdir(parents=True, exist_ok=True)
    print(f"[profiler] Traces: {trace_dir}")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    return profile(
        activities=activities,
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=args.profile_repeat),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=args.profile_stack,
    )


def move_batch_to_device(batch: Dict, device: torch.device, non_blocking: bool = False) -> Dict:
    """Move all tensors in batch to device."""
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device, non_blocking=non_blocking)
        elif isinstance(v, dict):
            result[k] = {kk: vv.to(device, non_blocking=non_blocking) for kk, vv in v.items()}
        else:
            result[k] = v
    return result


def compute_losses(outputs, batch, policy_weight=1.0, value_weight=0.5, time_weight=0.1,
                   value_cls_weight=0.3, start_square_weight=0.1, value_error_weight=0.1, 
                   promo_weight=0.1):
    """Compute losses for model outputs.
    
    Args:
        outputs: tuple of (move_logits [B, 4098], value_out [B, 1], 
                          value_cls_out [B, 3], value_error_out [B, 1],
                          time_cls_out [B, 256], start_square_logits [B, 64], (optional promo_logits))
        batch: dict with policy_target, y_val, y_val_cls, time_target_cls, legal_mask
    """
    if len(outputs) == 7:
        move_logits, value_out, value_cls_out, value_error_out, time_cls_out, start_square_logits, promo_logits = outputs
    else:
        move_logits, value_out, value_cls_out, value_error_out, time_cls_out, start_square_logits = outputs
        promo_logits = None
    
    # Policy loss: cross-entropy on 4098 classes
    policy_target = batch['policy_target']  # [B] int64 indices
    policy_loss = F.cross_entropy(move_logits, policy_target)
    
    # Promotion loss: cross-entropy on 4 classes (Q, R, B, N)
    # Compute for all samples to avoid data-dependent branching that breaks torch.compile
    if promo_logits is not None and 'promo_target' in batch and 'promo_file' in batch:
        promo_target = batch['promo_target']  # [B] 0=None, 1=Q, 2=R, 3=B, 4=N
        promo_file = batch['promo_file']      # [B] 0-7
        promo_type = (promo_target - 1).clamp(min=0)  # [B] 0=Q,1=R,2=B,3=N (clamp to handle promo_target=0)
        
        # Select logits for each sample's promo_file: (B, 8, 4) -> (B, 4)
        batch_indices = torch.arange(promo_logits.shape[0], device=promo_logits.device)
        selected_logits = promo_logits[batch_indices, promo_file, :]  # [B, 4]
        
        # Compute per-sample loss
        per_sample_loss = F.cross_entropy(selected_logits, promo_type, reduction='none')  # [B]
        
        # Mask: only count samples with actual promotions (promo_target > 0)
        promo_mask = (promo_target > 0).float()
        promo_loss = (per_sample_loss * promo_mask).sum() / (promo_mask.sum() + 1e-8)
    else:
        promo_loss = torch.tensor(0.0, device=move_logits.device)

    # Value regression loss: MSE on sigmoid output vs target (0=loss, 0.5=draw, 1=win)
    value_pred = torch.sigmoid(value_out.squeeze(-1))
    value_loss = F.mse_loss(value_pred, batch['y_val'])

    # Value error loss: predict squared error magnitude
    value_error_pred = value_error_out.squeeze(-1)
    value_error_target = (value_pred.detach() - batch['y_val']) ** 2
    value_error_loss = F.mse_loss(value_error_pred, value_error_target)
    
    # Value classification loss: cross-entropy on WDL (0=loss, 1=draw, 2=win)
    value_cls_loss = F.cross_entropy(value_cls_out, batch['y_val_cls'])
    
    # Time loss: cross-entropy on 256 bins (multimodal distribution)
    time_loss = F.cross_entropy(time_cls_out, batch['time_target_cls'])
    
    # Start square loss: cross-entropy on 64 squares
    # Target is the source square of the move (policy_target // 64 for regular moves)
    # For resign/flag (indices 4096, 4097), we mask them out without data-dependent branching
    regular_move_mask = (policy_target < 4096).float()
    # Clamp target to valid range for all samples (resign/flag would give invalid indices)
    start_square_target = (policy_target.clamp(max=4095) // 64).long()
    per_sample_start_loss = F.cross_entropy(start_square_logits, start_square_target, reduction='none')
    start_square_loss = (per_sample_start_loss * regular_move_mask).sum() / (regular_move_mask.sum() + 1e-8)
    
    total = (policy_weight * policy_loss + 
             value_weight * value_loss + 
             value_cls_weight * value_cls_loss +
             value_error_weight * value_error_loss +
             time_weight * time_loss +
             start_square_weight * start_square_loss +
             promo_weight * promo_loss)
    
    return total, {
        'policy': policy_loss.detach(),
        'value': value_loss.detach(),
        'value_cls': value_cls_loss.detach(),
        'value_error': value_error_loss.detach(),
        'time': time_loss.detach(),
        'start_sq': start_square_loss.detach(),
        'promo': promo_loss.detach(),
    }


def compute_metrics(outputs, batch) -> Dict[str, torch.Tensor]:
    """Compute accuracy metrics."""
    if len(outputs) == 7:
        move_logits, value_out, value_cls_out, value_error_out, time_cls_out, start_square_logits, _ = outputs
    else:
        move_logits, value_out, value_cls_out, value_error_out, time_cls_out, start_square_logits = outputs
    policy_target = batch['policy_target']
    
    # Policy accuracy (top-1 and top-5)
    # Note: model already masks illegal moves to -inf, so this is effectively legal_top1
    top1 = (move_logits.argmax(dim=1) == policy_target).float().mean()
    top5_preds = move_logits.topk(5, dim=1).indices  # [B, 5]
    top5 = (top5_preds == policy_target.unsqueeze(1)).any(dim=1).float().mean()
    
    # Value regression MAE
    value_pred = torch.sigmoid(value_out.squeeze(-1))
    value_mae = torch.abs(value_pred - batch['y_val']).mean()

    # Value error MAE
    value_error_pred = value_error_out.squeeze(-1)
    value_error_target = (value_pred.detach() - batch['y_val']) ** 2
    value_error_mae = torch.abs(value_error_pred - value_error_target).mean()
    
    # Value classification accuracy (WDL)
    value_cls_acc = (value_cls_out.argmax(dim=1) == batch['y_val_cls']).float().mean()
    
    # Time classification accuracy (256 bins)
    time_cls_acc = (time_cls_out.argmax(dim=1) == batch['time_target_cls']).float().mean()
    
    # Time top-5 accuracy (within 5 bins = ~2% of range)
    time_top5_preds = time_cls_out.topk(5, dim=1).indices
    time_top5 = (time_top5_preds == batch['time_target_cls'].unsqueeze(1)).any(dim=1).float().mean()
    
    # Time "close" accuracy: within 10 bins (~4% of range)
    time_pred_bin = time_cls_out.argmax(dim=1)
    time_close = (torch.abs(time_pred_bin - batch['time_target_cls']) <= 10).float().mean()
    
    # Start square accuracy (for regular moves only) - no data-dependent branching
    regular_move_mask = (policy_target < 4096).float()
    start_square_target = (policy_target.clamp(max=4095) // 64).long()
    start_square_correct = (start_square_logits.argmax(dim=1) == start_square_target).float()
    start_square_acc = (start_square_correct * regular_move_mask).sum() / (regular_move_mask.sum() + 1e-8)
    
    # Promotion accuracy - no data-dependent branching
    promo_acc = torch.tensor(0.0, device=move_logits.device)
    promo_count = torch.tensor(0.0, device=move_logits.device)
    
    if 'promo_target' in batch and len(outputs) == 7:
        promo_logits = outputs[6]
        promo_target = batch['promo_target']
        promo_file = batch['promo_file']
        promo_type = (promo_target - 1).clamp(min=0)  # Handle promo_target=0
        
        promo_mask = (promo_target > 0).float()
        promo_count = promo_mask.sum()
        
        # Select logits for each sample's promo_file: (B, 8, 4) -> (B, 4)
        batch_indices = torch.arange(promo_logits.shape[0], device=promo_logits.device)
        selected_logits = promo_logits[batch_indices, promo_file, :]
        
        promo_preds = selected_logits.argmax(dim=1)
        promo_correct = (promo_preds == promo_type).float()
        promo_acc = (promo_correct * promo_mask).sum() / (promo_mask.sum() + 1e-8)

    return {
        'policy_top1': top1.detach(),
        'policy_top5': top5.detach(),
        'value_mae': value_mae.detach(),
        'value_error_mae': value_error_mae.detach(),
        'value_cls_acc': value_cls_acc.detach(),
        'time_cls_acc': time_cls_acc.detach(),
        'time_top5': time_top5.detach(),
        'time_close': time_close.detach(),
        'start_sq_acc': start_square_acc.detach(),
        'promo_acc': promo_acc.detach(),
        'promo_count': promo_count.detach(),
    }


# ELO bins: 1200-1300, 1300-1400, ..., 2500-2600
ELO_BINS = list(range(1200, 2600, 100))  # [1200, 1300, ..., 2500]
ELO_BIN_NAMES = [f"{lo}-{lo+100}" for lo in ELO_BINS]


def evaluate(model, loader, device, non_blocking, max_batches=0, args=None) -> Dict[str, float]:
    """Evaluate model on validation data with per-ELO accuracy tracking."""
    model.eval()
    totals = defaultdict(float)
    sample_count = 0
    
    # Per-ELO bin accumulators
    num_bins = len(ELO_BINS)
    elo_top1_correct = torch.zeros(num_bins, device=device)
    elo_top5_correct = torch.zeros(num_bins, device=device)
    elo_counts = torch.zeros(num_bins, device=device)

    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch = move_batch_to_device(batch, device, non_blocking)
            
            outputs = model(batch, return_promo=True)
            loss, loss_terms = compute_losses(
                outputs, batch,
                policy_weight=args.policy_weight if args else 1.0,
                value_weight=args.value_weight if args else 0.5,
                value_error_weight=args.value_error_weight if args else 0.1,
                time_weight=args.time_weight if args else 0.5,
                value_cls_weight=args.value_cls_weight if args else 0.3,
                start_square_weight=args.start_square_weight if args else 0.1,
                promo_weight=args.promo_weight if args else 0.1,
            )
            metrics = compute_metrics(outputs, batch)
            
            batch_size = batch['board_history'].size(0)
            sample_count += batch_size
            totals['loss'] += loss.detach() * batch_size
            for name, value in {**loss_terms, **metrics}.items():
                totals[name] += value * batch_size
            
            # Per-ELO accuracy tracking
            move_logits = outputs[0]
            policy_target = batch['policy_target']
            
            # Get active_elo from scalars (index 0, normalized as (elo-1900)/700)
            # Denormalize: elo = scalars[:, 0] * 700 + 1900
            active_elo = batch['scalars'][:, 0] * 700 + 1900
            
            # Compute per-sample correctness
            top1_correct = (move_logits.argmax(dim=1) == policy_target).float()
            top5_correct = (move_logits.topk(5, dim=1).indices == policy_target.unsqueeze(1)).any(dim=1).float()
            
            # Bin the ELOs: clamp to [1200, 2599] then bin
            elo_clamped = active_elo.clamp(1200, 2599)
            bin_indices = ((elo_clamped - 1200) / 100).long().clamp(0, num_bins - 1)
            
            # Accumulate using scatter_add
            elo_top1_correct.scatter_add_(0, bin_indices, top1_correct)
            elo_top5_correct.scatter_add_(0, bin_indices, top5_correct)
            elo_counts.scatter_add_(0, bin_indices, torch.ones_like(top1_correct))

    results = {k: (v / sample_count).item() for k, v in totals.items()}
    
    # Add per-ELO metrics
    for i, bin_name in enumerate(ELO_BIN_NAMES):
        count = elo_counts[i].item()
        if count > 0:
            results[f"elo_{bin_name}_top1"] = (elo_top1_correct[i] / count).item()
            results[f"elo_{bin_name}_top5"] = (elo_top5_correct[i] / count).item()
            results[f"elo_{bin_name}_count"] = int(count)
    
    return results


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    global_step: int,
    args,
    non_blocking: bool,
    profiler=None,
    val_loader=None,
    best_val_loss: float = float("inf"),
    run_dir: Path = None,
):
    model.train()
    totals = defaultdict(float)
    sample_count = 0
    log_timer = time.perf_counter()
    last_log_step = global_step
    last_log_sample_count = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device, non_blocking)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(batch, return_promo=True)
            raw_loss, loss_terms = compute_losses(
                outputs, batch,
                policy_weight=args.policy_weight,
                value_weight=args.value_weight,
                value_error_weight=args.value_error_weight,
                time_weight=args.time_weight,
                value_cls_weight=args.value_cls_weight,
                start_square_weight=args.start_square_weight,
                promo_weight=args.promo_weight,
            )
            loss = raw_loss / args.grad_accum_steps

        scaler.scale(loss).backward()
        micro_step += 1

        should_step = micro_step % args.grad_accum_steps == 0
        if should_step:
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        metrics = compute_metrics(outputs, batch)
        batch_size = batch['board_history'].size(0)
        sample_count += batch_size
        totals['loss'] += raw_loss.detach() * batch_size
        for name, value in {**loss_terms, **metrics}.items():
            totals[name] += value * batch_size

        if should_step:
            global_step += 1
            if global_step % args.log_interval == 0:
                writer.add_scalar("train/total_loss", raw_loss.item(), global_step)
                for name, value in loss_terms.items():
                    writer.add_scalar(f"train/{name}_loss", value.item(), global_step)
                for name, value in metrics.items():
                    writer.add_scalar(f"train/{name}", value.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

                elapsed = max(1e-6, time.perf_counter() - log_timer)
                steps_since = max(1, global_step - last_log_step)
                samples_since = sample_count - last_log_sample_count
                it_per_sec = steps_since / elapsed
                samples_per_sec = samples_since / elapsed
                print(
                    f"[train] step {global_step:,} | {it_per_sec:.2f} it/s ({samples_per_sec:,.0f} samp/s) | loss {raw_loss.item():.3f} | "
                    f"policy {loss_terms['policy'].item():.3f} | value {loss_terms['value'].item():.4f} | "
                    f"top1 {metrics['policy_top1'].item():.3f}"
                )
                log_timer = time.perf_counter()
                last_log_step = global_step
                last_log_sample_count = sample_count

            # Mid-epoch validation
            if args.val_interval > 0 and val_loader is not None and global_step % args.val_interval == 0:
                print(f"\n[val] Running validation at step {global_step:,}...")
                val_stats = evaluate(model, val_loader, device, non_blocking, max_batches=args.val_batches, args=args)
                print(
                    f"[val] step {global_step:,} | loss {val_stats.get('loss', 0):.4f} | "
                    f"top1 {val_stats.get('policy_top1', 0):.3f} | "
                    f"val_cls {val_stats.get('value_cls_acc', 0):.3f} | time_cls {val_stats.get('time_cls_acc', 0):.3f}"
                )
                # Print per-ELO accuracy summary
                elo_keys = [k for k in val_stats if k.startswith("elo_") and k.endswith("_top1")]
                if elo_keys:
                    elo_summary = " | ".join(
                        f"{k.replace('elo_', '').replace('_top1', '')}:{val_stats[k]:.2f}"
                        for k in sorted(elo_keys)[:6]  # Show first 6 bins
                    )
                    print(f"[val] per-ELO top1: {elo_summary} ...")
                for name, value in val_stats.items():
                    writer.add_scalar(f"val/{name}", value, global_step)

                if val_stats['loss'] < best_val_loss:
                    best_val_loss = val_stats['loss']
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss,
                        Path(args.checkpoint_dir) / f"chessformer_{args.config}_best.pt",
                        args.config, run_dir
                    )

                model.train()
                log_timer = time.perf_counter()
                last_log_step = global_step

        if profiler is not None:
            profiler.step()

    # Handle remaining accumulated gradients
    if micro_step % args.grad_accum_steps != 0:
        if args.grad_clip and args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    epoch_stats = {k: (v / sample_count).item() for k, v in totals.items()}
    return epoch_stats, global_step, best_val_loss


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, path: Path, config_name: str, run_dir: Optional[Path] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "config": config_name,
        "version": "1.0",
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    if run_dir is not None:
        checkpoint["run_dir"] = str(run_dir)
    torch.save(checkpoint, path)
    print(f"[checkpoint] Saved to {path}")


def load_checkpoint(path: Path, model, optimizer, scheduler=None, scaler=None, device=None):
    print(f"[checkpoint] Loading from {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model_state = checkpoint["model"]
    try:
        model.load_state_dict(model_state)
    except RuntimeError:
        new_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
        model.load_state_dict(new_state)
    
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    
    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    run_dir = checkpoint.get("run_dir", None)
    
    print(f"[checkpoint] Resumed from epoch {epoch}, step {global_step}, best_val_loss {best_val_loss:.4f}")
    return epoch, global_step, best_val_loss, run_dir


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")

    if args.config == "large":
        config = copy.deepcopy(CONFIG_LARGE)
    else:  # small (default)
        config = copy.deepcopy(CONFIG_SMALL)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        config['gradient_checkpointing'] = True
    
    device = torch.device(args.device)
    configure_precision(device, args.disable_tf32)

    model = Chessformer(config).to(device=device)
    if args.compile_model:
        try:
            # Use 'default' mode when gradient accumulation is enabled to avoid instability
            # max-autotune can cause issues with gradient accumulation
            compile_mode = args.compile_mode
            if args.grad_accum_steps > 1 and compile_mode == "max-autotune":
                compile_mode = "default"
                print(f"[train] Switching compile mode to 'default' for gradient accumulation stability")
            model = torch.compile(model, mode=compile_mode)
            print(f"[train] torch.compile enabled with mode={compile_mode}")
        except Exception as exc:
            print(f"[train] torch.compile failed ({exc}); continuing without.")
    print(f"Using {args.config} config ({count_parameters(model):,} params, fp16 mixed precision)")

    # Create dataloaders
    pin_memory = device.type == "cuda"
    non_blocking = pin_memory
    
    train_loader = create_dataloader(
        args.data_dir, split='train',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    val_loader = create_dataloader(
        args.data_dir, split='val',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    train_samples = count_samples(args.data_dir, 'train')
    val_samples = count_samples(args.data_dir, 'val')
    print(f"[dataset] Train samples: ~{train_samples:,} | Val samples: ~{val_samples:,}")

    # Optimizer
    optimizer_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    # Scheduler
    approx_batches = max(1, math.ceil(train_samples / args.batch_size))
    steps_per_epoch = max(1, math.ceil(approx_batches / args.grad_accum_steps))
    total_steps = args.epochs * steps_per_epoch
    
    # Linear warmup + cosine decay
    warmup_steps = args.warmup_steps
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Cosine decay after warmup
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"[scheduler] Linear warmup ({warmup_steps} steps) + cosine decay")
    
    # Mixed precision: use GradScaler for fp16 training
    scaler = GradScaler(enabled=device.type == "cuda")

    # Resume state
    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    resumed_run_dir = None

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        start_epoch, global_step, best_val_loss, resumed_run_dir = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )
        start_epoch += 1

    # Run directory
    if args.resume_dir:
        run_dir = Path(args.resume_dir)
    elif resumed_run_dir and args.resume:
        run_dir = Path(resumed_run_dir)
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = Path(args.log_dir) / f"chessformer_{args.config}_{timestamp}"

    writer = SummaryWriter(run_dir)
    if start_epoch == 1:
        writer.add_text("config", str(config))

    profiler_context = create_profiler(args, run_dir, device, steps_per_epoch)
    with profiler_context as profiler:
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")

            # Ensure streaming parquet order changes each epoch (deterministically).
            if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            train_stats, global_step, best_val_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler, device, writer,
                epoch, global_step, args, non_blocking, profiler,
                val_loader=val_loader, best_val_loss=best_val_loss, run_dir=run_dir,
            )
            print(
                f"Train loss: {train_stats.get('loss', 0):.4f} | "
                f"top1 {train_stats.get('policy_top1', 0):.3f} | "
                f"val_cls {train_stats.get('value_cls_acc', 0):.3f} | "
                f"time_cls {train_stats.get('time_cls_acc', 0):.3f}"
            )

            val_stats = evaluate(model, val_loader, device, non_blocking, max_batches=args.val_batches, args=args)
            print(
                f"Val loss: {val_stats.get('loss', 0):.4f} | "
                f"top1 {val_stats.get('policy_top1', 0):.3f} | "
                f"val_cls {val_stats.get('value_cls_acc', 0):.3f} | "
                f"time_cls {val_stats.get('time_cls_acc', 0):.3f} | time_close {val_stats.get('time_close', 0):.3f}"
            )
            # Print per-ELO accuracy breakdown
            elo_keys = sorted([k for k in val_stats if k.startswith("elo_") and k.endswith("_top1")])
            if elo_keys:
                print("[val] per-ELO top1 accuracy:")
                for k in elo_keys:
                    bin_name = k.replace('elo_', '').replace('_top1', '')
                    count_key = f"elo_{bin_name}_count"
                    count = val_stats.get(count_key, 0)
                    print(f"  {bin_name}: {val_stats[k]:.3f} (n={count:,})")

            for name, value in train_stats.items():
                writer.add_scalar(f"train_epoch/{name}", value, epoch)
            for name, value in val_stats.items():
                writer.add_scalar(f"val/{name}", value, epoch)

            ckpt_path = Path(args.checkpoint_dir) / f"chessformer_{args.config}_epoch{epoch:02d}.pt"
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, ckpt_path, args.config, run_dir)
            
            if val_stats['loss'] < best_val_loss:
                best_val_loss = val_stats['loss']
                best_path = Path(args.checkpoint_dir) / f"chessformer_{args.config}_best.pt"
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, best_path, args.config, run_dir)

    writer.close()
    print(f"\n[done] Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
