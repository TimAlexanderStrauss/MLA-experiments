"""Train one cell of the MoE-backbone 2x2 attention ablation."""

import argparse
import csv
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from model import ATTENTION_MODES, GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MoE 2x2 condition")
    parser.add_argument("--attn_mode", choices=ATTENTION_MODES, default="mha")
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--mla_d_c", type=int, default=128)
    parser.add_argument("--mla_d_c_q", type=int, default=192)
    parser.add_argument("--mla_d_rope", type=int, default=32)
    parser.add_argument("--first_dense_layers", type=int, default=1)
    parser.add_argument("--n_shared_experts", type=int, default=2)
    parser.add_argument("--n_routed_experts", type=int, default=16)
    parser.add_argument("--num_experts_per_tok", type=int, default=2)
    parser.add_argument("--moe_intermediate_size", type=int, default=336)
    parser.add_argument("--aux_loss_alpha", type=float, default=0.001)
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
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=600)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--data_dir", default="../data/fineweb_edu")
    parser.add_argument("--out_dir", default="results/run")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=False
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
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)


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
            torch.from_numpy(data[index : index + block_size].astype(np.int64))
            for index in starts
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                data[index + 1 : index + block_size + 1].astype(np.int64)
            )
            for index in starts
        ]
    )
    return x.to(device), y.to(device)


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda" or dtype == torch.float32:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate(
    model: GPT,
    val_data: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    precision: torch.dtype,
    val_generator: torch.Generator,
) -> tuple[float, float, list[dict[str, float | int]]]:
    model.eval()
    ce_losses: list[float] = []
    auxiliary_losses: list[float] = []
    aggregate: dict[int, dict[str, torch.Tensor | int]] = {}
    for _ in range(args.eval_iters):
        x, y = get_batch(
            val_data, args.block_size, args.batch_size, device, val_generator
        )
        with _autocast(device, precision):
            _, cross_entropy, auxiliary_loss, routing_stats = model(x, y)
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

    routing_rows: list[dict[str, float | int]] = []
    for layer, stats in sorted(aggregate.items()):
        token_count = int(stats["token_count"])
        counts = stats["selected_counts"].float().cpu()
        probabilities = stats["probability_sums"].float().cpu() / token_count
        fractions = counts / (token_count * args.num_experts_per_tok)
        entropy = float(stats["entropy_sum"].float().cpu() / token_count)
        for expert in range(args.n_routed_experts):
            routing_rows.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "selected_fraction": float(fractions[expert]),
                    "mean_probability": float(probabilities[expert]),
                    "router_entropy": entropy,
                }
            )
    model.train()
    return (
        float(np.mean(ce_losses)),
        float(np.mean(auxiliary_losses)),
        routing_rows,
    )


def routing_summary(
    rows: list[dict[str, float | int]], n_experts: int
) -> tuple[float, float, float]:
    if not rows:
        return float("nan"), float("nan"), float("nan")
    entropies = {int(row["layer"]): float(row["router_entropy"]) for row in rows}
    loads_by_layer: dict[int, list[float]] = {}
    for row in rows:
        loads_by_layer.setdefault(int(row["layer"]), []).append(
            float(row["selected_fraction"])
        )
    max_load = max(max(loads) for loads in loads_by_layer.values())
    cvs = []
    for loads in loads_by_layer.values():
        array = np.asarray(loads)
        cvs.append(float(array.std(ddof=0) / (1.0 / n_experts)))
    return float(np.mean(list(entropies.values()))), max_load, float(np.mean(cvs))


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
    ignored = {"out_dir", "data_dir", "save_interval"}
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_generator = torch.Generator(device="cpu").manual_seed(
        args.train_data_seed
    )
    val_generator = torch.Generator(device="cpu").manual_seed(args.val_data_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_dir = Path(args.data_dir)
    train_path, val_path = data_dir / "train.bin", data_dir / "val.bin"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing {train_path} or {val_path}; see README.md")
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")

    config = GPTConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        attn_mode=args.attn_mode,
        mla_d_c=args.mla_d_c,
        mla_d_c_q=args.mla_d_c_q,
        mla_d_rope=args.mla_d_rope,
        first_dense_layers=args.first_dense_layers,
        n_shared_experts=args.n_shared_experts,
        n_routed_experts=args.n_routed_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        moe_intermediate_size=args.moe_intermediate_size,
        aux_loss_alpha=args.aux_loss_alpha,
    )
    model = GPT(config).to(device)
    print(
        f"Mode={args.attn_mode} total_params={model.count_parameters():,} "
        f"active_params/token={model.count_active_parameters():,}"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "float16")
    optimizer = model.configure_optimizers(
        args.weight_decay, args.lr, (args.beta1, args.beta2), device.type
    )

    checkpoint_path = out_dir / "checkpoint.pt"
    metrics_path = out_dir / "metrics.csv"
    routing_path = out_dir / "routing.csv"
    start_iter = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        validate_resume_config(checkpoint["config"], args, checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # ``map_location=device`` puts every saved tensor on the GPU, but RNG
        # states must be restored from CPU ByteTensors.
        train_generator.set_state(checkpoint["train_generator_state"].cpu())
        val_generator.set_state(checkpoint["val_generator_state"].cpu())
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            )
        start_iter = int(checkpoint["iter"]) + 1
        print(f"Resuming from {checkpoint_path} at iteration {start_iter}")

    if args.compile:
        model = torch.compile(model)

    config_path = out_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(
            json.dumps(vars(args), indent=2), encoding="utf-8"
        )
    metrics_exists = metrics_path.exists() and metrics_path.stat().st_size > 0
    routing_exists = routing_path.exists() and routing_path.stat().st_size > 0
    metrics_file = metrics_path.open("a", newline="", encoding="utf-8")
    routing_file = routing_path.open("a", newline="", encoding="utf-8")
    metrics_writer = csv.DictWriter(
        metrics_file,
        fieldnames=[
            "iter",
            "train_loss",
            "optimization_loss",
            "train_aux_loss",
            "val_loss",
            "val_aux_loss",
            "router_entropy",
            "max_load_fraction",
            "load_cv",
            "lr",
            "tokens_seen",
            "elapsed_s",
            "attn_mode",
            "seed",
            "train_data_seed",
            "val_data_seed",
        ],
    )
    routing_writer = csv.DictWriter(
        routing_file,
        fieldnames=[
            "iter",
            "attn_mode",
            "seed",
            "layer",
            "expert",
            "selected_fraction",
            "mean_probability",
            "router_entropy",
        ],
    )
    if not metrics_exists:
        metrics_writer.writeheader()
    if not routing_exists:
        routing_writer.writeheader()

    tokens_per_iter = args.batch_size * args.grad_accum * args.block_size
    started = time.time()
    model.train()

    def log_evaluation(
        iteration: int,
        train_ce: float,
        train_total: float,
        train_aux: float,
        learning_rate: float,
    ) -> None:
        val_ce, val_aux, rows = evaluate(
            model, val_data, args, device, precision, val_generator
        )
        # Evaluation runs under ``no_grad`` and therefore allocates blocks of
        # very different shapes than the training graph does. On Windows/WDDM
        # every cached CUDA block is also backed by a system commit charge, so
        # holding the eval-shaped blocks across the eval->train boundary both
        # fragments VRAM and inflates commit at the exact moment the training
        # step needs its activation memory back. Returning them to the driver
        # here is numerically inert and costs a few hundred ms per eval.
        if device.type == "cuda":
            torch.cuda.empty_cache()
        entropy, max_load, load_cv = routing_summary(
            rows, args.n_routed_experts
        )
        for row in rows:
            routing_writer.writerow(
                {
                    "iter": iteration,
                    "attn_mode": args.attn_mode,
                    "seed": args.seed,
                    **row,
                }
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
                "lr": learning_rate,
                "tokens_seen": iteration * tokens_per_iter,
                "elapsed_s": round(time.time() - started, 1),
                "attn_mode": args.attn_mode,
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
            f"max_load={max_load:.3f}"
        )

    if start_iter == 0:
        log_evaluation(
            0,
            float("nan"),
            float("nan"),
            float("nan"),
            get_lr(0, args.warmup_iters, args.max_iters, args.lr, args.min_lr),
        )

    # Rate is measured over this process's own iterations. Using the wall clock
    # since ``started`` divided by ``iteration`` would report a nonsensical rate
    # after a resume, where a large iteration count pairs with a small elapsed.
    loop_started = time.time()
    for step in range(start_iter, args.max_iters):
        learning_rate = get_lr(
            step, args.warmup_iters, args.max_iters, args.lr, args.min_lr
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        cross_entropy_accumulator = 0.0
        auxiliary_accumulator = 0.0
        total_accumulator = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(
                train_data,
                args.block_size,
                args.batch_size,
                device,
                train_generator,
            )
            with _autocast(device, precision):
                _, cross_entropy, auxiliary_loss, _ = model(
                    x, y, collect_routing_stats=False
                )
                assert cross_entropy is not None
                total_loss = cross_entropy + auxiliary_loss
                scaled_loss = total_loss / args.grad_accum
            # ``.detach()`` before the scalar conversion: these are logging-only
            # reads of tensors that are still attached to the autograd graph,
            # which PyTorch warns about explicitly. Numerically identical.
            cross_entropy_accumulator += (
                float(cross_entropy.detach()) / args.grad_accum
            )
            auxiliary_accumulator += (
                float(auxiliary_loss.detach()) / args.grad_accum
            )
            total_accumulator += float(total_loss.detach()) / args.grad_accum
            scaler.scale(scaled_loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        iteration = step + 1
        # Heartbeat. Without it the only output is one line per ``eval_interval``
        # (500) iterations, so a wedged run and a healthy one look identical for
        # many minutes. Reserved VRAM is printed because this job runs close to
        # the card's capacity and creeping reservation is the failure to watch.
        if args.log_interval > 0 and iteration % args.log_interval == 0:
            reserved = (
                torch.cuda.memory_reserved() / 1024**3
                if device.type == "cuda"
                else 0.0
            )
            print(
                f"iter {iteration:5d}/{args.max_iters} "
                f"loss={total_accumulator:.4f} lr={learning_rate:.2e} "
                f"{(time.time() - loop_started) / (iteration - start_iter):.2f}s/it "
                f"vram_reserved={reserved:.2f}GiB",
                flush=True,
            )
        if iteration % args.eval_interval == 0:
            log_evaluation(
                iteration,
                cross_entropy_accumulator,
                total_accumulator,
                auxiliary_accumulator,
                learning_rate,
            )
        if iteration % args.save_interval == 0 or iteration == args.max_iters:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                step,
                args,
                train_generator,
                val_generator,
            )

    metrics_file.close()
    routing_file.close()
    print(
        f"Done: {args.attn_mode} seed={args.seed}; "
        f"elapsed={(time.time() - started) / 3600:.2f}h"
    )


if __name__ == "__main__":
    main()
