"""
Training script for the MLA 2×2 ablation study.

Adapted from nanoGPT (Karpathy) with the following changes:
  - No learned position embeddings (RoPE handles position in attention)
  - GPTConfig.attn_mode selects one of 4 attention variants
  - CSV logging for downstream ANOVA analysis
  - Checkpoint save/resume for multi-day runs
  - Seed argument for reproducibility

Usage (on the desktop PC):
  python train.py --attn_mode=mha       --seed=42  --out_dir=results/mha_s42
  python train.py --attn_mode=mha_rope  --seed=42  --out_dir=results/mha_rope_s42
  python train.py --attn_mode=mla_norope--seed=42  --out_dir=results/mla_norope_s42
  python train.py --attn_mode=mla       --seed=42  --out_dir=results/mla_s42

Data: expects data/fineweb_edu/train.bin and data/fineweb_edu/val.bin as uint16 numpy arrays.
Run data/fineweb_edu/prepare.py first to create them.
"""

import argparse
import csv
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GPT ablation variant")
    # Architecture
    p.add_argument("--attn_mode",    default="mha",
                   choices=["mha", "mha_rope", "mla_norope", "mla"])
    p.add_argument("--n_layer",      type=int,   default=6)
    p.add_argument("--n_head",       type=int,   default=8)
    p.add_argument("--n_embd",       type=int,   default=512)
    p.add_argument("--block_size",   type=int,   default=512)
    p.add_argument("--mla_d_c",      type=int,   default=128)
    p.add_argument("--mla_d_c_q",    type=int,   default=192)
    p.add_argument("--mla_d_rope",   type=int,   default=32)
    # Training
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--max_iters",    type=int,   default=15300,
                   help="15300 iters x 32768 tokens/iter ~= 501M tokens at eff. batch=64, seq=512")
    p.add_argument("--batch_size",   type=int,   default=16,
                   help="Micro-batch size per gradient accumulation step")
    p.add_argument("--grad_accum",   type=int,   default=4,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--lr",           type=float, default=6e-4)
    p.add_argument("--min_lr",       type=float, default=6e-5)
    p.add_argument("--warmup_iters", type=int,   default=2000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1",        type=float, default=0.9)
    p.add_argument("--beta2",        type=float, default=0.95)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    # Evaluation & checkpointing
    p.add_argument("--eval_interval",  type=int, default=500)
    p.add_argument("--eval_iters",     type=int, default=100,
                   help="Number of micro-batches to evaluate on")
    p.add_argument("--save_interval",  type=int, default=5000)
    # Paths
    p.add_argument("--data_dir",  default="data/fineweb_edu")
    p.add_argument("--out_dir",   default="results/run")
    # System
    p.add_argument("--dtype",     default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--compile",   action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile the model (use --no-compile to disable)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# LR schedule (cosine with linear warmup)
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup: int, max_iters: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * step / warmup
    if step >= max_iters:
        return min_lr
    decay = (step - warmup) / (max_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return min_lr + coeff * (lr - min_lr)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64)) for i in ix
    ])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: GPT,
    val_data: np.ndarray,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
    ctx,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(val_data, block_size, batch_size, device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    use_compile = args.compile
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Determinism ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Device / dtype ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ptdtype = {"float32": torch.float32, "float16": torch.float16,
               "bfloat16": torch.bfloat16}[args.dtype]
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=ptdtype)
        if device.type == "cuda" else nullcontext()
    )

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---- Data ----
    data_dir = Path(args.data_dir)
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data   = np.memmap(data_dir / "val.bin",   dtype=np.uint16, mode="r")
    print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

    # ---- Model ----
    config = GPTConfig(
        block_size = args.block_size,
        vocab_size = 50257,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        n_embd     = args.n_embd,
        dropout    = 0.0,
        bias       = False,
        attn_mode  = args.attn_mode,
        mla_d_c    = args.mla_d_c,
        mla_d_c_q  = args.mla_d_c_q,
        mla_d_rope = args.mla_d_rope,
    )

    # Resume from checkpoint if present
    ckpt_path   = out_dir / "checkpoint.pt"
    metrics_path = out_dir / "metrics.csv"
    start_iter  = 0

    model = GPT(config).to(device)
    print(f"Parameters: {model.count_parameters():,}  |  attn_mode: {args.attn_mode}")

    # GradScaler only for fp16 (bfloat16 does not need it)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16"))

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=(args.beta1, args.beta2),
        device_type=device.type,
    ) if hasattr(model, "configure_optimizers") else torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter"] + 1
        print(f"  -> Resuming at iter {start_iter}")

    if use_compile:
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile: failed ({e}), running without")

    # ---- CSV logging ----
    csv_mode = "a" if ckpt_path.exists() else "w"
    csv_file = open(metrics_path, csv_mode, newline="")
    csv_writer = csv.writer(csv_file)
    if csv_mode == "w":
        csv_writer.writerow([
            "iter", "train_loss", "val_loss", "lr",
            "tokens_seen", "elapsed_s", "attn_mode", "seed",
        ])
        csv_file.flush()

    # Save config once
    config_path = out_dir / "config.json"
    if not config_path.exists():
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)

    # ---- Training loop ----
    effective_batch = args.batch_size * args.grad_accum
    tokens_per_iter = effective_batch * args.block_size
    print(f"Effective batch size: {effective_batch}  |  Tokens/iter: {tokens_per_iter:,}")
    print(f"Total iters: {args.max_iters}  |  Total tokens: ~{args.max_iters * tokens_per_iter / 1e6:.0f}M")

    t_start = time.time()
    tokens_seen = start_iter * tokens_per_iter
    model.train()

    # Initial eval
    if start_iter == 0:
        val_loss = evaluate(model, val_data, args.block_size, args.batch_size,
                            args.eval_iters, device, ctx)
        print(f"[iter 0] val_loss={val_loss:.4f}")
        lr0 = get_lr(0, args.warmup_iters, args.max_iters, args.lr, args.min_lr)
        csv_writer.writerow([0, float("nan"), val_loss, lr0, 0, 0.0,
                             args.attn_mode, args.seed])
        csv_file.flush()

    for step in range(start_iter, args.max_iters):
        lr = get_lr(step, args.warmup_iters, args.max_iters, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        train_loss_accum = 0.0
        for micro_step in range(args.grad_accum):
            x, y = get_batch(train_data, args.block_size, args.batch_size, device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            train_loss_accum += loss.item()
            scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        tokens_seen += tokens_per_iter

        # ---- Evaluation ----
        if (step + 1) % args.eval_interval == 0:
            val_loss = evaluate(model, val_data, args.block_size, args.batch_size,
                                args.eval_iters, device, ctx)
            elapsed = time.time() - t_start
            tokens_M = tokens_seen / 1e6
            print(
                f"[iter {step+1:5d}/{args.max_iters}]  "
                f"train={train_loss_accum:.4f}  val={val_loss:.4f}  "
                f"lr={lr:.2e}  tok={tokens_M:.0f}M  t={elapsed/3600:.1f}h"
            )
            csv_writer.writerow([
                step + 1, train_loss_accum, val_loss, lr,
                tokens_seen, round(elapsed, 1), args.attn_mode, args.seed,
            ])
            csv_file.flush()

        # ---- Checkpoint ----
        if (step + 1) % args.save_interval == 0 or (step + 1) == args.max_iters:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(
                {
                    "iter":      step,
                    "model":     raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config":    vars(args),
                },
                ckpt_path,
            )
            print(f"  Checkpoint saved at iter {step+1}")

    csv_file.close()
    total_time = time.time() - t_start
    print(f"\nDone. Total time: {total_time/3600:.2f}h  |  {args.attn_mode} seed={args.seed}")


if __name__ == "__main__":
    main()
