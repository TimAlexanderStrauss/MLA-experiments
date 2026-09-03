"""Train one cell of the DeepSeek-near 2x2x2 mechanism experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from model import ATTENTION_MODES, BACKBONES, GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attn_mode", choices=ATTENTION_MODES, required=True)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--mla_d_c", type=int, default=256)
    parser.add_argument("--mla_d_rope", type=int, default=32)
    parser.add_argument("--first_dense_layers", type=int, default=1)
    parser.add_argument("--n_shared_experts", type=int, default=2)
    parser.add_argument("--n_routed_experts", type=int, default=16)
    parser.add_argument("--num_experts_per_tok", type=int, default=2)
    parser.add_argument("--moe_intermediate_size", type=int, default=336)
    parser.add_argument("--dense_intermediate_size", type=int, default=1344)
    parser.add_argument("--aux_loss_alpha", type=float, default=0.001)
    parser.add_argument("--moe_dispatch", choices=("loop", "batched"), default="batched")
    parser.add_argument("--moe_capacity_factor", type=float, default=2.25)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_seed", type=int)
    parser.add_argument("--val_data_seed", type=int)
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
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=600)
    parser.add_argument("--routing_eval_iters", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--data_dir", type=Path, default=Path("../data/fineweb_edu"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--compile_mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    return parser.parse_args()


def get_lr(
    step: int, warmup: int, max_iters: int, lr: float, min_lr: float
) -> float:
    if step < warmup:
        return lr * step / warmup
    if step >= max_iters:
        return min_lr
    ratio = (step - warmup) / (max_iters - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (lr - min_lr)


def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(
        len(data) - block_size - 1,
        (batch_size,),
        generator=generator,
        device="cpu",
    ).tolist()
    tokens = np.empty((batch_size, block_size + 1), dtype=np.int64)
    for row, start in enumerate(starts):
        tokens[row] = data[start : start + block_size + 1]
    tensor = torch.from_numpy(tokens)
    if device.type == "cuda":
        tensor = tensor.pin_memory().to(device, non_blocking=True)
    else:
        tensor = tensor.to(device)
    return tensor[:, :-1], tensor[:, 1:]


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda" or dtype == torch.float32:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def make_config(args: argparse.Namespace) -> GPTConfig:
    return GPTConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        attn_mode=args.attn_mode,
        backbone=args.backbone,
        mla_d_c=args.mla_d_c,
        mla_d_rope=args.mla_d_rope,
        first_dense_layers=args.first_dense_layers,
        n_shared_experts=args.n_shared_experts,
        n_routed_experts=args.n_routed_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        moe_intermediate_size=args.moe_intermediate_size,
        dense_intermediate_size=args.dense_intermediate_size,
        aux_loss_alpha=args.aux_loss_alpha,
        moe_dispatch=args.moe_dispatch,
        moe_capacity_factor=args.moe_capacity_factor,
    )


@torch.no_grad()
def evaluate(
    model: GPT,
    val_data: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    precision: torch.dtype,
    generator: torch.Generator,
) -> tuple[float, float, list[dict[str, float | int]]]:
    model.eval()
    ce_losses = []
    auxiliary_losses = []
    aggregate: dict[int, dict[str, torch.Tensor | int]] = {}
    for evaluation_step in range(args.eval_iters):
        x, y = get_batch(
            val_data, args.block_size, args.batch_size, device, generator
        )
        collect = (
            args.backbone == "moe"
            and evaluation_step < min(args.routing_eval_iters, args.eval_iters)
        )
        with autocast_context(device, precision):
            _, cross_entropy, auxiliary_loss, routing_stats = model(
                x, y, collect_routing_stats=collect
            )
        assert cross_entropy is not None
        ce_losses.append(float(cross_entropy))
        auxiliary_losses.append(float(auxiliary_loss))
        for stats in routing_stats:
            layer = int(stats["layer"])
            if layer not in aggregate:
                aggregate[layer] = {
                    "selected_counts": stats["selected_counts"].clone(),
                    "probability_sums": stats["probability_sums"].clone(),
                    "entropy_sum": stats["entropy_sum"].clone(),
                    "token_count": int(stats["token_count"]),
                }
            else:
                current = aggregate[layer]
                current["selected_counts"] += stats["selected_counts"]
                current["probability_sums"] += stats["probability_sums"]
                current["entropy_sum"] += stats["entropy_sum"]
                current["token_count"] += int(stats["token_count"])

    rows: list[dict[str, float | int]] = []
    for layer, stats in sorted(aggregate.items()):
        token_count = int(stats["token_count"])
        counts = stats["selected_counts"].float().cpu()
        probabilities = stats["probability_sums"].float().cpu() / token_count
        fractions = counts / (token_count * args.num_experts_per_tok)
        entropy = float(stats["entropy_sum"].float().cpu() / token_count)
        for expert in range(args.n_routed_experts):
            rows.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "selected_fraction": float(fractions[expert]),
                    "mean_probability": float(probabilities[expert]),
                    "router_entropy": entropy,
                }
            )
    model.train()
    return float(np.mean(ce_losses)), float(np.mean(auxiliary_losses)), rows


def routing_summary(
    rows: list[dict[str, float | int]], n_experts: int
) -> tuple[float, float, float]:
    if not rows:
        return float("nan"), float("nan"), float("nan")
    entropy_by_layer = {
        int(row["layer"]): float(row["router_entropy"]) for row in rows
    }
    loads_by_layer: dict[int, list[float]] = {}
    for row in rows:
        loads_by_layer.setdefault(int(row["layer"]), []).append(
            float(row["selected_fraction"])
        )
    maximum = max(max(values) for values in loads_by_layer.values())
    cvs = [
        float(np.std(values) / (1.0 / n_experts))
        for values in loads_by_layer.values()
    ]
    return float(np.mean(list(entropy_by_layer.values()))), maximum, float(np.mean(cvs))


def raw_model(model: GPT) -> GPT:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def serializable_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    train_generator: torch.Generator,
    val_generator: torch.Generator,
    elapsed_s: float,
) -> None:
    checkpoint = {
        "iter": step,
        "model": raw_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": serializable_args(args),
        "train_generator_state": train_generator.get_state(),
        "val_generator_state": val_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "elapsed_s": elapsed_s,
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint, path)


def validate_resume_config(
    saved: dict, current: argparse.Namespace, checkpoint_path: Path
) -> None:
    ignored = {"out_dir", "data_dir", "save_interval", "log_interval"}
    changed = {
        key: (value, getattr(current, key, None))
        for key, value in saved.items()
        if key not in ignored and value != getattr(current, key, None)
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
    if args.batch_size * args.grad_accum != 64:
        raise ValueError("batch_size * grad_accum must equal 64")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_generator = torch.Generator(device="cpu").manual_seed(args.train_data_seed)
    val_generator = torch.Generator(device="cpu").manual_seed(args.val_data_seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Production runs require the RTX GPU. "
            "Use --device=cpu only for a small smoke test."
        )
    device = torch.device(args.device)
    precision = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    train_path = args.data_dir / "train.bin"
    val_path = args.data_dir / "val.bin"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing {train_path} or {val_path}")
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")

    model = GPT(make_config(args)).to(device)
    print(
        f"backbone={args.backbone} attention={args.attn_mode} "
        f"params={model.count_parameters():,} "
        f"active={model.count_active_parameters():,} "
        f"logical_kv_cache={model.logical_kv_cache_elements_per_token():,}"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "float16")
    optimizer = model.configure_optimizers(
        args.weight_decay, args.lr, (args.beta1, args.beta2), device.type
    )

    checkpoint_path = args.out_dir / "checkpoint.pt"
    metrics_path = args.out_dir / "metrics.csv"
    routing_path = args.out_dir / "routing.csv"
    start_iter = 0
    checkpoint_elapsed_s: float | None = None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        validate_resume_config(checkpoint["config"], args, checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_generator.set_state(checkpoint["train_generator_state"].cpu())
        val_generator.set_state(checkpoint["val_generator_state"].cpu())
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            )
        start_iter = int(checkpoint["iter"]) + 1
        if "elapsed_s" in checkpoint:
            checkpoint_elapsed_s = float(checkpoint["elapsed_s"])
        print(f"Resume at iteration {start_iter}")

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("This PyTorch build does not provide torch.compile")
        model = torch.compile(
            model, mode=args.compile_mode, fullgraph=False, dynamic=False
        )

    config_path = args.out_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(
            json.dumps(serializable_args(args), indent=2), encoding="utf-8"
        )
    runtime = {
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "cuda_version": torch.version.cuda,
    }
    runtime_path = args.out_dir / "runtime.json"
    if runtime_path.exists():
        saved_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if saved_runtime != runtime:
            raise ValueError(
                f"Runtime changed for resumed run: saved={saved_runtime}, "
                f"current={runtime}"
            )
    else:
        runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    metrics_exists = metrics_path.exists() and metrics_path.stat().st_size > 0
    routing_exists = routing_path.exists() and routing_path.stat().st_size > 0
    metrics_fields = [
        "iter", "train_loss", "optimization_loss", "train_aux_loss",
        "val_loss", "val_aux_loss", "router_entropy", "max_load_fraction",
        "load_cv", "lr", "tokens_seen", "elapsed_s", "tokens_per_second",
        "peak_vram_gib", "attn_mode", "backbone", "seed",
        "train_data_seed", "val_data_seed",
    ]
    routing_fields = [
        "iter", "attn_mode", "backbone", "seed", "layer", "expert",
        "selected_fraction", "mean_probability", "router_entropy",
    ]
    elapsed_offset_s = checkpoint_elapsed_s or 0.0
    if metrics_exists and checkpoint_elapsed_s is None:
        with metrics_path.open(newline="", encoding="utf-8") as existing_file:
            elapsed_values = []
            for row in csv.DictReader(existing_file):
                try:
                    row_iteration = int(row["iter"])
                    if not checkpoint_path.exists() or row_iteration <= start_iter:
                        elapsed_values.append(float(row["elapsed_s"]))
                except (KeyError, TypeError, ValueError):
                    pass
            if elapsed_values:
                elapsed_offset_s = max(elapsed_values)

    metrics_file = metrics_path.open("a", newline="", encoding="utf-8")
    routing_file = routing_path.open("a", newline="", encoding="utf-8")
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=metrics_fields)
    routing_writer = csv.DictWriter(routing_file, fieldnames=routing_fields)
    if not metrics_exists:
        metrics_writer.writeheader()
    if not routing_exists:
        routing_writer.writeheader()

    tokens_per_iteration = args.batch_size * args.grad_accum * args.block_size
    started = time.perf_counter()
    loop_started = time.perf_counter()
    model.train()

    def log_evaluation(
        iteration: int, train_ce: float, train_total: float, train_aux: float, lr: float
    ) -> None:
        val_ce, val_aux, rows = evaluate(
            model, val_data, args, device, precision, val_generator
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        entropy, max_load, load_cv = routing_summary(rows, args.n_routed_experts)
        for row in rows:
            routing_writer.writerow(
                {
                    "iter": iteration,
                    "attn_mode": args.attn_mode,
                    "backbone": args.backbone,
                    "seed": args.seed,
                    **row,
                }
            )
        elapsed = elapsed_offset_s + time.perf_counter() - started
        peak_vram = (
            torch.cuda.max_memory_reserved() / 1024**3
            if device.type == "cuda" else 0.0
        )
        metrics_writer.writerow(
            {
                "iter": iteration,
                "train_loss": train_ce,
                "optimization_loss": train_total,
                "train_aux_loss": train_aux,
                "val_loss": val_ce,
                "val_aux_loss": val_aux,
                "router_entropy": entropy,
                "max_load_fraction": max_load,
                "load_cv": load_cv,
                "lr": lr,
                "tokens_seen": iteration * tokens_per_iteration,
                "elapsed_s": round(elapsed, 1),
                "tokens_per_second": (
                    iteration * tokens_per_iteration / elapsed if iteration else float("nan")
                ),
                "peak_vram_gib": peak_vram,
                "attn_mode": args.attn_mode,
                "backbone": args.backbone,
                "seed": args.seed,
                "train_data_seed": args.train_data_seed,
                "val_data_seed": args.val_data_seed,
            }
        )
        metrics_file.flush()
        routing_file.flush()
        print(
            f"[iter {iteration:5d}] train_ce={train_ce:.4f} "
            f"val_ce={val_ce:.4f} aux={val_aux:.5f} "
            f"max_load={max_load:.3f}",
            flush=True,
        )

    if start_iter == 0:
        log_evaluation(
            0, float("nan"), float("nan"), float("nan"),
            get_lr(0, args.warmup_iters, args.max_iters, args.lr, args.min_lr),
        )

    for step in range(start_iter, args.max_iters):
        learning_rate = get_lr(
            step, args.warmup_iters, args.max_iters, args.lr, args.min_lr
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        ce_accumulator = aux_accumulator = total_accumulator = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(
                train_data, args.block_size, args.batch_size, device, train_generator
            )
            with autocast_context(device, precision):
                _, cross_entropy, auxiliary_loss, _ = model(
                    x, y, collect_routing_stats=False
                )
                assert cross_entropy is not None
                total_loss = cross_entropy + auxiliary_loss
                scaled_loss = total_loss / args.grad_accum
            ce_accumulator += float(cross_entropy.detach()) / args.grad_accum
            aux_accumulator += float(auxiliary_loss.detach()) / args.grad_accum
            total_accumulator += float(total_loss.detach()) / args.grad_accum
            scaler.scale(scaled_loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        iteration = step + 1
        if args.log_interval > 0 and iteration % args.log_interval == 0:
            elapsed_loop = time.perf_counter() - loop_started
            completed = iteration - start_iter
            peak = (
                torch.cuda.max_memory_reserved() / 1024**3
                if device.type == "cuda" else 0.0
            )
            print(
                f"iter {iteration:5d}/{args.max_iters} loss={total_accumulator:.4f} "
                f"lr={learning_rate:.2e} {elapsed_loop / completed:.3f}s/it "
                f"{completed * tokens_per_iteration / elapsed_loop:,.0f} tok/s "
                f"peak_vram={peak:.2f}GiB",
                flush=True,
            )
        if iteration % args.eval_interval == 0:
            log_evaluation(
                iteration, ce_accumulator, total_accumulator, aux_accumulator, learning_rate
            )
        if iteration % args.save_interval == 0 or iteration == args.max_iters:
            save_checkpoint(
                checkpoint_path, model, optimizer, step, args,
                train_generator, val_generator,
                elapsed_offset_s + time.perf_counter() - started,
            )

    metrics_file.close()
    routing_file.close()
    (args.out_dir / "DONE").touch()
    print(
        f"Done: {args.backbone}/{args.attn_mode} seed={args.seed}; "
        f"elapsed={(time.perf_counter() - started) / 3600:.2f}h"
    )


if __name__ == "__main__":
    main()
