"""Train one condition of the DeepSeek-layout sensitivity study.

Training and validation batches use dedicated CPU ``torch.Generator`` objects.
Their seeds depend only on the experimental seed, not on model architecture.
Consequently, all three conditions see exactly the same sampled token windows
for a given seed. Generator states are checkpointed for resume correctness.
"""

import argparse
import csv
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a paper-layout follow-up condition"
    )
    parser.add_argument(
        "--attn_mode",
        default="mha",
        choices=["mha", "mla_current", "mla_deepseek"],
    )
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--mla_d_c", type=int, default=128)
    parser.add_argument("--mla_d_c_q", type=int, default=192)
    parser.add_argument("--mla_d_rope", type=int, default=32)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_seed", type=int, default=None)
    parser.add_argument("--val_data_seed", type=int, default=None)
    parser.add_argument("--max_iters", type=int, default=15300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=600)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--data_dir", default="../data/fineweb_edu")
    parser.add_argument("--out_dir", default="results/run")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the model. Disabled in the planned study for comparability.",
    )
    return parser.parse_args()


def get_lr(
    step: int, warmup: int, max_iters: int, lr: float, min_lr: float
) -> float:
    if step < warmup:
        return lr * step / warmup
    if step >= max_iters:
        return min_lr
    decay_ratio = (step - warmup) / (max_iters - warmup)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coefficient * (lr - min_lr)


def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(
        len(data) - block_size,
        (batch_size,),
        generator=generator,
        device="cpu",
    ).tolist()
    x = torch.stack(
        [
            torch.from_numpy(data[i : i + block_size].astype(np.int64))
            for i in starts
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64))
            for i in starts
        ]
    )
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(
    model: GPT,
    val_data: np.ndarray,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
    autocast_context,
    val_generator: torch.Generator,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(
            val_data,
            block_size,
            batch_size,
            device,
            val_generator,
        )
        with autocast_context:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    train_generator: torch.Generator,
    val_generator: torch.Generator,
) -> None:
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "iter": step,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": vars(args),
        "train_generator_state": train_generator.get_state(),
        "val_generator_state": val_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint, path)


def validate_resume_config(
    saved: dict, current: argparse.Namespace, checkpoint_path: Path
) -> None:
    """Refuse a resume that silently changes a controlled setting."""
    controlled = [
        "attn_mode",
        "n_layer",
        "n_head",
        "n_embd",
        "block_size",
        "mla_d_c",
        "mla_d_c_q",
        "mla_d_rope",
        "seed",
        "train_data_seed",
        "val_data_seed",
        "max_iters",
        "batch_size",
        "grad_accum",
        "lr",
        "min_lr",
        "warmup_iters",
        "weight_decay",
        "beta1",
        "beta2",
        "grad_clip",
        "eval_interval",
        "eval_iters",
        "dtype",
        "compile",
    ]
    changed = {
        key: (saved.get(key), getattr(current, key))
        for key in controlled
        if saved.get(key) != getattr(current, key)
    }
    if changed:
        details = ", ".join(
            f"{key}: saved={old!r}, current={new!r}"
            for key, (old, new) in changed.items()
        )
        raise ValueError(
            f"Refusing incompatible resume from {checkpoint_path}: {details}"
        )


def main() -> None:
    args = parse_args()
    if args.train_data_seed is None:
        args.train_data_seed = 100_000 + args.seed
    if args.val_data_seed is None:
        args.val_data_seed = 200_000 + args.seed

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # These generators are deliberately independent of model initialization.
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(args.train_data_seed)
    val_generator = torch.Generator(device="cpu")
    val_generator.manual_seed(args.val_data_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    precision = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    autocast_context = (
        torch.amp.autocast(device_type="cuda", dtype=precision)
        if device.type == "cuda"
        else nullcontext()
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path} or {val_path}. See README.md, section 'Daten'."
        )
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=50257,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
        bias=False,
        attn_mode=args.attn_mode,
        mla_d_c=args.mla_d_c,
        mla_d_c_q=args.mla_d_c_q,
        mla_d_rope=args.mla_d_rope,
    )
    model = GPT(config).to(device)
    print(
        f"Parameters: {model.count_parameters():,} | mode: {args.attn_mode} | "
        f"model seed: {args.seed} | train data seed: {args.train_data_seed} | "
        f"val data seed: {args.val_data_seed}"
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16"))
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=(args.beta1, args.beta2),
        device_type=device.type,
    )

    checkpoint_path = out_dir / "checkpoint.pt"
    metrics_path = out_dir / "metrics.csv"
    start_iter = 0
    if checkpoint_path.exists():
        print(f"Resuming from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        validate_resume_config(checkpoint["config"], args, checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_generator.set_state(checkpoint["train_generator_state"])
        val_generator.set_state(checkpoint["val_generator_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        start_iter = checkpoint["iter"] + 1
        print(f"Resuming at iteration {start_iter}")

    if args.compile:
        model = torch.compile(model)
        print("torch.compile: enabled")

    csv_mode = "a" if checkpoint_path.exists() else "w"
    csv_file = metrics_path.open(csv_mode, newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if csv_mode == "w":
        writer.writerow(
            [
                "iter",
                "train_loss",
                "val_loss",
                "lr",
                "tokens_seen",
                "elapsed_s",
                "attn_mode",
                "seed",
                "train_data_seed",
                "val_data_seed",
            ]
        )
        csv_file.flush()

    config_path = out_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(
            json.dumps(vars(args), indent=2), encoding="utf-8"
        )

    effective_batch = args.batch_size * args.grad_accum
    tokens_per_iter = effective_batch * args.block_size
    print(
        f"Effective batch: {effective_batch} | tokens/iter: {tokens_per_iter:,} | "
        f"total tokens: {args.max_iters * tokens_per_iter:,}"
    )

    start_time = time.time()
    tokens_seen = start_iter * tokens_per_iter
    model.train()

    if start_iter == 0:
        val_loss = evaluate(
            model,
            val_data,
            args.block_size,
            args.batch_size,
            args.eval_iters,
            device,
            autocast_context,
            val_generator,
        )
        initial_lr = get_lr(
            0, args.warmup_iters, args.max_iters, args.lr, args.min_lr
        )
        writer.writerow(
            [
                0,
                float("nan"),
                val_loss,
                initial_lr,
                0,
                0.0,
                args.attn_mode,
                args.seed,
                args.train_data_seed,
                args.val_data_seed,
            ]
        )
        csv_file.flush()
        print(f"[iter 0] val_loss={val_loss:.4f}")

    for step in range(start_iter, args.max_iters):
        learning_rate = get_lr(
            step, args.warmup_iters, args.max_iters, args.lr, args.min_lr
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(
                train_data,
                args.block_size,
                args.batch_size,
                device,
                train_generator,
            )
            with autocast_context:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            train_loss += loss.item()
            scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        tokens_seen += tokens_per_iter

        if (step + 1) % args.eval_interval == 0:
            val_loss = evaluate(
                model,
                val_data,
                args.block_size,
                args.batch_size,
                args.eval_iters,
                device,
                autocast_context,
                val_generator,
            )
            elapsed = time.time() - start_time
            print(
                f"[iter {step + 1:5d}/{args.max_iters}] "
                f"train={train_loss:.4f} val={val_loss:.4f} "
                f"lr={learning_rate:.2e} tok={tokens_seen / 1e6:.0f}M "
                f"t={elapsed / 3600:.2f}h"
            )
            writer.writerow(
                [
                    step + 1,
                    train_loss,
                    val_loss,
                    learning_rate,
                    tokens_seen,
                    round(elapsed, 1),
                    args.attn_mode,
                    args.seed,
                    args.train_data_seed,
                    args.val_data_seed,
                ]
            )
            csv_file.flush()

        if (step + 1) % args.save_interval == 0 or step + 1 == args.max_iters:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                step,
                args,
                train_generator,
                val_generator,
            )
            print(f"Checkpoint saved at iteration {step + 1}")

    csv_file.close()
    print(
        f"Done: {args.attn_mode}, seed={args.seed}, "
        f"session time={(time.time() - start_time) / 3600:.2f}h"
    )


if __name__ == "__main__":
    main()
