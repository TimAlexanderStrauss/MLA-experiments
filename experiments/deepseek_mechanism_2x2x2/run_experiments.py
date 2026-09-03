"""Run the balanced 24-cell production plan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from model import ATTENTION_MODES, BACKBONES


SEEDS = (42, 123, 456)
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
BASE_CONDITIONS = (
    ("dense", "mha"),
    ("moe", "mla_decoupled"),
    ("dense", "mla_coupled"),
    ("moe", "mha_decoupled"),
    ("dense", "mha_decoupled"),
    ("moe", "mla_coupled"),
    ("dense", "mla_decoupled"),
    ("moe", "mha"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("gpu_profile.json"))
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--data_dir", type=Path, default=Path("../data/fineweb_edu"))
    parser.add_argument("--backbone", choices=BACKBONES)
    parser.add_argument("--attn_mode", choices=ATTENTION_MODES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def balanced_plan() -> list[tuple[str, str, int]]:
    plan = []
    for seed_index, seed in enumerate(SEEDS):
        shift = (seed_index * 3) % len(BASE_CONDITIONS)
        conditions = BASE_CONDITIONS[shift:] + BASE_CONDITIONS[:shift]
        plan.extend((backbone, attention, seed) for backbone, attention in conditions)
    return plan


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def main() -> None:
    args = parse_args()
    if not args.profile.exists():
        raise FileNotFoundError(
            f"Missing {args.profile}. Run benchmark_gpu.py before production."
        )
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    required = {
        "device_name", "torch_version", "cuda_version", "batch_size",
        "grad_accum", "compile", "moe_dispatch",
    }
    if missing := required - profile.keys():
        raise ValueError(f"GPU profile is missing: {sorted(missing)}")
    if int(profile["batch_size"]) * int(profile["grad_accum"]) != 64:
        raise ValueError("GPU profile does not preserve effective batch size 64")
    if profile.get("architecture") != ARCHITECTURE:
        raise ValueError(
            "GPU profile does not match the registered architecture. "
            "Run benchmark_gpu.py again."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Refusing to start production runs.")
    actual_device = torch.cuda.get_device_name(0)
    if actual_device != profile["device_name"]:
        raise RuntimeError(
            f"GPU profile was created for {profile['device_name']!r}, "
            f"but the active GPU is {actual_device!r}. Run benchmark_gpu.py again."
        )
    if str(torch.__version__) != str(profile["torch_version"]):
        raise RuntimeError("PyTorch changed after the GPU benchmark. Create a new profile.")
    if torch.version.cuda != profile["cuda_version"]:
        raise RuntimeError("CUDA changed after the GPU benchmark. Create a new profile.")
    if not (args.data_dir / "train.bin").exists() or not (args.data_dir / "val.bin").exists():
        raise FileNotFoundError(f"FineWeb-Edu files are missing below {args.data_dir}")

    selected = []
    for backbone, attention, seed in balanced_plan():
        if args.backbone and backbone != args.backbone:
            continue
        if args.attn_mode and attention != args.attn_mode:
            continue
        if args.seed and seed != args.seed:
            continue
        selected.append((backbone, attention, seed))

    ran = skipped = 0
    for backbone, attention, seed in selected:
        out_dir = args.results_dir / f"{backbone}_{attention}_s{seed}"
        if (out_dir / "DONE").exists():
            log(f"SKIP {backbone}/{attention} seed={seed} (DONE exists)")
            skipped += 1
            continue
        command = [
            sys.executable,
            "train.py",
            f"--backbone={backbone}",
            f"--attn_mode={attention}",
            f"--seed={seed}",
            f"--out_dir={out_dir}",
            f"--data_dir={args.data_dir}",
            f"--batch_size={profile['batch_size']}",
            f"--grad_accum={profile['grad_accum']}",
            f"--moe_dispatch={profile['moe_dispatch']}",
            f"--moe_capacity_factor={profile.get('moe_capacity_factor', 2.25)}",
            f"--compile_mode={profile.get('compile_mode', 'reduce-overhead')}",
            *[f"--{name}={value}" for name, value in ARCHITECTURE.items()],
            "--max_iters=15300",
            "--eval_interval=500",
            "--eval_iters=600",
            "--routing_eval_iters=50",
            "--save_interval=1000",
            "--dtype=bfloat16",
            "--device=cuda",
            "--compile" if profile["compile"] else "--no-compile",
        ]
        log(f"START {backbone}/{attention} seed={seed}")
        if args.dry_run:
            print(subprocess.list2cmdline(command))
            continue
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise SystemExit(completed.returncode)
        if not (out_dir / "DONE").exists():
            raise RuntimeError(f"Run ended without DONE flag: {out_dir}")
        log(f"DONE {backbone}/{attention} seed={seed}")
        ran += 1
    log(f"Finished. ran={ran} skipped={skipped}")


if __name__ == "__main__":
    main()
