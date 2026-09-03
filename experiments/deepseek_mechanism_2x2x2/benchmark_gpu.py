"""Select a fast and safe RTX training profile for the full experiment."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig
from train import get_batch


ARCHITECTURE = {
    "n_layer": 6,
    "n_head": 8,
    "n_embd": 512,
    "block_size": 512,
    "mla_d_c": 256,
    "mla_d_rope": 32,
    "first_dense_layers": 1,
    "n_shared_experts": 2,
    "n_routed_experts": 16,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 336,
    "dense_intermediate_size": 1344,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("gpu_profile.json"))
    parser.add_argument(
        "--data_dir", type=Path, default=Path("../data/fineweb_edu")
    )
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--timed_steps", type=int, default=3)
    parser.add_argument("--max_vram_fraction", type=float, default=0.90)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Test all batch, compile, and dispatch combinations.",
    )
    return parser.parse_args()


def candidates(full: bool) -> list[dict]:
    if full:
        return [
            {"batch_size": batch, "compile": compile_model, "moe_dispatch": dispatch}
            for batch in (32, 16, 8)
            for compile_model in (True, False)
            for dispatch in ("batched", "loop")
        ]
    return [
        {"batch_size": 32, "compile": True, "moe_dispatch": "batched"},
        {"batch_size": 16, "compile": True, "moe_dispatch": "batched"},
        {"batch_size": 32, "compile": False, "moe_dispatch": "batched"},
        {"batch_size": 16, "compile": False, "moe_dispatch": "batched"},
        {"batch_size": 16, "compile": False, "moe_dispatch": "loop"},
        {"batch_size": 8, "compile": False, "moe_dispatch": "batched"},
    ]


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()


def run_trial(
    spec: dict,
    warmup_steps: int,
    timed_steps: int,
    train_data: np.ndarray,
) -> dict:
    cleanup()
    batch_size = int(spec["batch_size"])
    grad_accum = 64 // batch_size
    result = {
        **spec,
        "grad_accum": grad_accum,
        "status": "failed",
    }
    try:
        config = GPTConfig(
            attn_mode="mla_decoupled",
            backbone="moe",
            moe_dispatch=str(spec["moe_dispatch"]),
            **ARCHITECTURE,
        )
        model = GPT(config).cuda()
        optimizer = model.configure_optimizers(
            0.1, 6e-4, (0.9, 0.95), "cuda"
        )
        if spec["compile"]:
            model = torch.compile(
                model, mode="reduce-overhead", fullgraph=False, dynamic=False
            )
        data_generator = torch.Generator(device="cpu").manual_seed(100_042)

        def step() -> None:
            optimizer.zero_grad(set_to_none=True)
            for _ in range(grad_accum):
                x, y = get_batch(
                    train_data,
                    config.block_size,
                    batch_size,
                    torch.device("cuda"),
                    data_generator,
                )
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, ce, auxiliary, _ = model(x, y, collect_routing_stats=False)
                    assert ce is not None
                    loss = (ce + auxiliary) / grad_accum
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        for _ in range(warmup_steps):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(timed_steps):
            step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        result.update(
            {
                "status": "ok",
                "seconds_per_optimizer_step": elapsed / timed_steps,
                "tokens_per_second": (
                    timed_steps * 64 * config.block_size / elapsed
                ),
                "peak_vram_gib": torch.cuda.max_memory_reserved() / 1024**3,
            }
        )
        del optimizer, model
    except Exception as error:  # benchmark must continue after OOM/compiler errors
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        cleanup()
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3:
        raise RuntimeError("At least 8 GiB of CUDA memory is required")
    train_path = args.data_dir / "train.bin"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path}. The benchmark includes the production data path."
        )
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    results = []
    for index, spec in enumerate(candidates(args.full), start=1):
        print(f"[{index}] {spec}", flush=True)
        result = run_trial(
            spec, args.warmup_steps, args.timed_steps, train_data
        )
        results.append(result)
        if result["status"] == "ok":
            print(
                f"    {result['tokens_per_second']:,.0f} tok/s, "
                f"{result['peak_vram_gib']:.2f} GiB",
                flush=True,
            )
        else:
            print(f"    FAILED: {result['error']}", flush=True)

    total_vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    successful = [
        result for result in results
        if result["status"] == "ok"
        and result["peak_vram_gib"] <= total_vram_gib * args.max_vram_fraction
    ]
    if not successful:
        raise RuntimeError(
            "No GPU profile completed below the VRAM safety limit. "
            "Run with smaller batches or increase --max_vram_fraction carefully."
        )
    best = max(successful, key=lambda item: item["tokens_per_second"])
    profile = {
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "total_vram_gib": total_vram_gib,
        "max_vram_fraction": args.max_vram_fraction,
        "effective_batch_size": 64,
        "batch_size": best["batch_size"],
        "grad_accum": best["grad_accum"],
        "compile": best["compile"],
        "compile_mode": "reduce-overhead",
        "moe_dispatch": best["moe_dispatch"],
        "moe_capacity_factor": 2.25,
        "architecture": ARCHITECTURE,
        "benchmark_tokens_per_second": best["tokens_per_second"],
        "benchmark_peak_vram_gib": best["peak_vram_gib"],
        "trials": results,
    }
    args.out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Selected profile: {args.out}")
    print(json.dumps({key: profile[key] for key in (
        "batch_size", "grad_accum", "compile", "moe_dispatch",
        "benchmark_tokens_per_second", "benchmark_peak_vram_gib",
    )}, indent=2))


if __name__ == "__main__":
    main()
