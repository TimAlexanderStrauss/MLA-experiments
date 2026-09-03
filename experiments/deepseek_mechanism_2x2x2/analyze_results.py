"""Validate and analyze the seed-paired DeepSeek-near 2x2x2 study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from model import ATTENTION_FACTORS, ATTENTION_MODES, BACKBONES, GPT, GPTConfig


SEEDS = (42, 123, 456)
FINAL_N_POINTS = 5
LABELS = {
    "mha": "MHA / coupled RoPE",
    "mha_decoupled": "MHA / decoupled RoPE",
    "mla_coupled": "Low-rank KV / coupled RoPE",
    "mla_decoupled": "DeepSeek-near MLA",
}
COLORS = {
    "mha": "#4c72b0",
    "mha_decoupled": "#55a868",
    "mla_coupled": "#c44e52",
    "mla_decoupled": "#dd8452",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--out_dir", type=Path, default=Path("plots"))
    return parser.parse_args()


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_frames = []
    routing_frames = []
    configs = []
    for run_dir in sorted(results_dir.glob("*")):
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.csv"
        if not run_dir.is_dir() or not config_path.exists() or not metrics_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        runtime_path = run_dir / "runtime.json"
        if runtime_path.exists():
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            config.update(
                {
                    "runtime_torch_version": runtime.get("torch_version"),
                    "runtime_device": runtime.get("device"),
                    "runtime_device_name": runtime.get("device_name"),
                    "runtime_cuda_version": runtime.get("cuda_version"),
                }
            )
        attention = config["attn_mode"]
        backbone = config["backbone"]
        seed = int(config["seed"])
        if attention not in ATTENTION_MODES or backbone not in BACKBONES:
            continue
        metrics = pd.read_csv(metrics_path).drop_duplicates("iter", keep="last")
        metrics["attn_mode"] = attention
        metrics["backbone"] = backbone
        metrics["seed"] = seed
        metric_frames.append(metrics)
        routing_path = run_dir / "routing.csv"
        if routing_path.exists() and routing_path.stat().st_size:
            routing = pd.read_csv(routing_path).drop_duplicates(
                ["iter", "layer", "expert"], keep="last"
            )
            if not routing.empty:
                routing_frames.append(routing)
        config.update({"run_dir": str(run_dir), "done": (run_dir / "DONE").exists()})
        configs.append(config)
    if not metric_frames:
        raise FileNotFoundError(f"No results found below {results_dir}")
    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.DataFrame(configs),
        pd.concat(routing_frames, ignore_index=True) if routing_frames else pd.DataFrame(),
    )


def validate_design(configs: pd.DataFrame) -> list[str]:
    notes = []
    keys = ["backbone", "attn_mode", "seed"]
    if configs.duplicated(keys).any():
        raise ValueError("Duplicate backbone/attention/seed cells detected")
    expected = {
        (backbone, attention, seed)
        for backbone in BACKBONES
        for attention in ATTENTION_MODES
        for seed in SEEDS
    }
    observed = set(
        zip(configs.backbone, configs.attn_mode, configs.seed.astype(int))
    )
    if missing := sorted(expected - observed):
        raise ValueError(f"Missing runs: {missing}")
    if extra := sorted(observed - expected):
        raise ValueError(f"Unexpected runs: {extra}")
    controlled = [
        "n_layer", "n_head", "n_embd", "block_size", "mla_d_c",
        "mla_d_rope", "first_dense_layers", "n_shared_experts",
        "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size",
        "dense_intermediate_size", "aux_loss_alpha", "moe_dispatch",
        "moe_capacity_factor", "max_iters", "batch_size", "grad_accum",
        "lr", "min_lr", "warmup_iters", "weight_decay", "eval_interval",
        "eval_iters", "routing_eval_iters", "dtype", "device", "compile", "compile_mode",
        "runtime_torch_version", "runtime_device", "runtime_device_name",
        "runtime_cuda_version",
    ]
    for field in controlled:
        if field not in configs:
            raise ValueError(f"Missing controlled config field: {field}")
        if configs[field].nunique(dropna=False) != 1:
            raise ValueError(f"Controlled config field differs: {field}")
    if not np.all(configs.batch_size * configs.grad_accum == 64):
        raise ValueError("Effective batch size is not 64 in all runs")
    for seed, group in configs.groupby("seed"):
        for field in ("train_data_seed", "val_data_seed"):
            if group[field].nunique() != 1:
                raise ValueError(f"{field} differs within seed {seed}")
    incomplete = configs.loc[~configs.done, "run_dir"].tolist()
    if incomplete:
        raise ValueError(f"Runs without DONE: {incomplete}")
    return notes


def compute_final_losses(
    metrics: pd.DataFrame, configs: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    notes = []
    lookup = {
        (row.backbone, row.attn_mode, int(row.seed)): row
        for row in configs.itertuples(index=False)
    }
    for key, group in metrics.groupby(["backbone", "attn_mode", "seed"]):
        config = lookup[(key[0], key[1], int(key[2]))]
        expected_last_eval = (
            int(config.max_iters) // int(config.eval_interval)
        ) * int(config.eval_interval)
        valid = group.dropna(subset=["val_loss"]).sort_values("iter")
        if valid.empty or int(valid.iter.max()) < expected_last_eval:
            notes.append(f"Incomplete metrics: {key}")
            continue
        final_points = valid.tail(FINAL_N_POINTS)
        if len(final_points) != FINAL_N_POINTS:
            notes.append(f"Fewer than five final evaluations: {key}")
            continue
        low_rank, decoupled = ATTENTION_FACTORS[key[1]]
        rows.append(
            {
                "backbone": key[0],
                "attn_mode": key[1],
                "seed": int(key[2]),
                "low_rank": low_rank,
                "decoupled_rope": decoupled,
                "final_val_loss": final_points.val_loss.mean(),
                "last_eval_iter": int(valid.iter.max()),
            }
        )
    return pd.DataFrame(rows), notes


def summarize(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for backbone in BACKBONES:
        for attention in ATTENTION_MODES:
            values = final.loc[
                (final.backbone == backbone) & (final.attn_mode == attention),
                "final_val_loss",
            ]
            n = len(values)
            mean = values.mean() if n else float("nan")
            standard_deviation = values.std(ddof=1) if n > 1 else float("nan")
            ci95 = (
                stats.t.ppf(0.975, n - 1) * standard_deviation / math.sqrt(n)
                if n > 1 else float("nan")
            )
            rows.append(
                {
                    "backbone": backbone,
                    "attn_mode": attention,
                    "label": LABELS[attention],
                    "n": n,
                    "mean": mean,
                    "std": standard_deviation,
                    "ci95": ci95,
                    "perplexity": math.exp(mean) if n else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _paired_test(name: str, values: pd.Series, definition: str) -> dict:
    values = values.dropna()
    n = len(values)
    mean = values.mean() if n else float("nan")
    standard_deviation = values.std(ddof=1) if n > 1 else float("nan")
    if n > 1:
        test = stats.ttest_1samp(values, 0.0)
        half_width = stats.t.ppf(0.975, n - 1) * standard_deviation / math.sqrt(n)
        t_statistic = float(test.statistic)
        p_value = float(test.pvalue)
        f_statistic = t_statistic**2
        partial_eta_squared = f_statistic / (f_statistic + n - 1)
    else:
        half_width = t_statistic = p_value = f_statistic = partial_eta_squared = float("nan")
    return {
        "effect": name,
        "definition": definition,
        "n_pairs": n,
        "mean_difference": mean,
        "std_difference": standard_deviation,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
        "t_statistic": t_statistic,
        "df_error": n - 1,
        "f_statistic": f_statistic,
        "p_value": p_value,
        "partial_eta_sq": partial_eta_squared,
    }


def compute_factorial_effects(final: pd.DataFrame) -> pd.DataFrame:
    pivot = final.pivot(
        index="seed", columns=["backbone", "attn_mode"], values="final_val_loss"
    )
    required = {(backbone, mode) for backbone in BACKBONES for mode in ATTENTION_MODES}
    if not required.issubset(set(pivot.columns)):
        return pd.DataFrame()

    def cell(backbone: str, low_rank: bool, decoupled: bool) -> pd.Series:
        mode = next(
            mode for mode, factors in ATTENTION_FACTORS.items()
            if factors == (low_rank, decoupled)
        )
        return pivot[(backbone, mode)]

    values: dict[tuple[bool, bool, bool], pd.Series] = {}
    for low_rank in (False, True):
        for decoupled in (False, True):
            for moe in (False, True):
                values[(low_rank, decoupled, moe)] = cell(
                    "moe" if moe else "dense", low_rank, decoupled
                )

    effects = [
        ("Low-rank KV", (True, False, False), 4),
        ("Decoupled RoPE", (False, True, False), 4),
        ("MoE backbone", (False, False, True), 4),
        ("Low-rank KV x Decoupled RoPE", (True, True, False), 2),
        ("Low-rank KV x MoE", (True, False, True), 2),
        ("Decoupled RoPE x MoE", (False, True, True), 2),
        ("Low-rank KV x Decoupled RoPE x MoE", (True, True, True), 1),
    ]
    rows = []
    for name, included, divisor in effects:
        contrast = None
        for factors, series in values.items():
            sign = 1
            for factor_value, use_factor in zip(factors, included):
                if use_factor:
                    sign *= 1 if factor_value else -1
            term = sign * series / divisor
            contrast = term if contrast is None else contrast + term
        rows.append(
            _paired_test(
                name,
                contrast,
                "seed-paired factorial contrast; positive means higher validation loss",
            )
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        order = frame.p_value.sort_values().index
        adjusted = pd.Series(index=frame.index, dtype=float)
        running = 0.0
        number = len(frame)
        for rank, index in enumerate(order):
            running = max(running, (number - rank) * frame.loc[index, "p_value"])
            adjusted.loc[index] = min(1.0, running)
        frame["p_value_holm"] = adjusted
    return frame


def compute_backbone_effects(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for backbone in BACKBONES:
        pivot = final[final.backbone == backbone].pivot(
            index="seed", columns="attn_mode", values="final_val_loss"
        )
        if not set(ATTENTION_MODES).issubset(pivot.columns):
            continue
        low_rank = (
            (pivot.mla_coupled + pivot.mla_decoupled)
            - (pivot.mha + pivot.mha_decoupled)
        ) / 2
        decoupled = (
            (pivot.mha_decoupled + pivot.mla_decoupled)
            - (pivot.mha + pivot.mla_coupled)
        ) / 2
        interaction = (
            pivot.mla_decoupled - pivot.mla_coupled
            - pivot.mha_decoupled + pivot.mha
        )
        for name, values in (
            ("Low-rank KV", low_rank),
            ("Decoupled RoPE", decoupled),
            ("Low-rank KV x Decoupled RoPE", interaction),
        ):
            row = _paired_test(name, values, "within-backbone seed-paired contrast")
            row["backbone"] = backbone
            rows.append(row)
    return pd.DataFrame(rows)


def compute_simple_contrasts(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for backbone in BACKBONES:
        pivot = final[final.backbone == backbone].pivot(
            index="seed", columns="attn_mode", values="final_val_loss"
        )
        if not set(ATTENTION_MODES).issubset(pivot.columns):
            continue
        contrasts = {
            "Decoupled RoPE without low-rank KV": pivot.mha_decoupled - pivot.mha,
            "Decoupled RoPE with low-rank KV": pivot.mla_decoupled - pivot.mla_coupled,
            "Low-rank KV with coupled RoPE": pivot.mla_coupled - pivot.mha,
            "Low-rank KV with decoupled RoPE": pivot.mla_decoupled - pivot.mha_decoupled,
            "Full MLA versus MHA": pivot.mla_decoupled - pivot.mha,
        }
        for name, values in contrasts.items():
            row = _paired_test(name, values, "within-backbone seed-paired simple contrast")
            row["backbone"] = backbone
            rows.append(row)
    return pd.DataFrame(rows)


def architecture_table(configs: pd.DataFrame) -> pd.DataFrame:
    first = configs.iloc[0]
    rows = []
    for backbone in BACKBONES:
        for attention in ATTENTION_MODES:
            model = GPT(
                GPTConfig(
                    block_size=int(first.block_size),
                    n_layer=int(first.n_layer),
                    n_head=int(first.n_head),
                    n_embd=int(first.n_embd),
                    attn_mode=attention,
                    backbone=backbone,
                    mla_d_c=int(first.mla_d_c),
                    mla_d_rope=int(first.mla_d_rope),
                    first_dense_layers=int(first.first_dense_layers),
                    n_shared_experts=int(first.n_shared_experts),
                    n_routed_experts=int(first.n_routed_experts),
                    num_experts_per_tok=int(first.num_experts_per_tok),
                    moe_intermediate_size=int(first.moe_intermediate_size),
                    dense_intermediate_size=int(first.dense_intermediate_size),
                    aux_loss_alpha=float(first.aux_loss_alpha),
                    moe_dispatch=str(first.moe_dispatch),
                    moe_capacity_factor=float(first.moe_capacity_factor),
                )
            )
            rows.append(
                {
                    "backbone": backbone,
                    "attn_mode": attention,
                    "total_parameters": model.count_parameters(),
                    "active_parameters_per_token": model.count_active_parameters(),
                    "logical_kv_cache_elements_per_token": model.logical_kv_cache_elements_per_token(),
                    "logical_kv_cache_with_recompute_per_token": (
                        model.logical_kv_cache_elements_with_recompute_per_token()
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_learning_curves(metrics: pd.DataFrame, out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for axis, backbone in zip(axes, BACKBONES):
        subset = metrics[metrics.backbone == backbone]
        for mode in ATTENTION_MODES:
            values = subset[subset.attn_mode == mode].groupby("iter").val_loss.agg(["mean", "std"])
            axis.plot(values.index, values["mean"], color=COLORS[mode], label=LABELS[mode])
            axis.fill_between(
                values.index,
                values["mean"] - values["std"].fillna(0),
                values["mean"] + values["std"].fillna(0),
                color=COLORS[mode], alpha=0.15,
            )
        axis.set(title=f"{backbone.upper()} backbone", xlabel="Iteration", ylabel="Validation CE")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("DeepSeek-near 2x2x2 learning curves")
    figure.tight_layout()
    figure.savefig(out_dir / "learning_curves.png", dpi=180)
    plt.close(figure)


def plot_heatmaps(summary: pd.DataFrame, out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    all_means = summary["mean"].to_numpy(float)
    low, high = np.nanmin(all_means), np.nanmax(all_means)
    for axis, backbone in zip(axes, BACKBONES):
        lookup = summary[summary.backbone == backbone].set_index("attn_mode")["mean"]
        matrix = np.array([
            [lookup.mha, lookup.mha_decoupled],
            [lookup.mla_coupled, lookup.mla_decoupled],
        ])
        image = axis.imshow(matrix, cmap="RdYlGn_r", vmin=low, vmax=high)
        for row in range(2):
            for column in range(2):
                axis.text(column, row, f"{matrix[row, column]:.4f}", ha="center", va="center")
        axis.set_xticks((0, 1), ("Coupled", "Decoupled"))
        axis.set_yticks((0, 1), ("Full-rank KV", "Low-rank KV"))
        axis.set_title(backbone.upper())
        axis.set_xlabel("RoPE path")
    axes[0].set_ylabel("KV projection")
    figure.colorbar(image, ax=axes, label="Final validation CE", shrink=0.85)
    figure.suptitle("Seed means for all eight cells")
    figure.savefig(out_dir / "heatmaps_2x2x2.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_final_losses(final: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 6))
    x = np.arange(len(ATTENTION_MODES))
    width = 0.36
    for offset, backbone in zip((-width / 2, width / 2), BACKBONES):
        subset = summary[summary.backbone == backbone].set_index("attn_mode")
        means = [subset.loc[mode, "mean"] for mode in ATTENTION_MODES]
        cis = [subset.loc[mode, "ci95"] for mode in ATTENTION_MODES]
        axis.bar(x + offset, means, width, yerr=cis, capsize=4, label=backbone.upper(), alpha=0.8)
        for seed in SEEDS:
            points = final[(final.backbone == backbone) & (final.seed == seed)].set_index("attn_mode")
            if set(ATTENTION_MODES).issubset(points.index):
                axis.plot(x + offset, [points.loc[mode, "final_val_loss"] for mode in ATTENTION_MODES], color="black", alpha=0.18)
    axis.set_xticks(x, [LABELS[mode] for mode in ATTENTION_MODES], rotation=8)
    axis.set_ylabel("Final validation CE")
    axis.set_title("DeepSeek-near 2x2x2 final loss")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_dir / "final_val_loss.png", dpi=180)
    plt.close(figure)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "Study incomplete."
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = [f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    backbone_effects: pd.DataFrame,
    simple_contrasts: pd.DataFrame,
    architecture: pd.DataFrame,
    notes: list[str],
) -> None:
    lines = [
        "# Generated results: DeepSeek-near 2x2x2",
        "",
        "Validation loss contains cross-entropy only. Positive effects mean higher loss.",
        "",
        "## Eight condition means",
        "",
        markdown_table(summary),
        "",
        "## Seed-paired 2x2x2 effects",
        "",
        markdown_table(effects),
        "",
        "## Seed-paired effects within each backbone",
        "",
        markdown_table(backbone_effects),
        "",
        "## Mechanism-relevant simple contrasts",
        "",
        markdown_table(simple_contrasts),
        "",
        "## Architecture and logical cache",
        "",
        markdown_table(architecture),
        "",
        "## Validation notes",
        "",
    ]
    lines.extend([f"- {note}" for note in notes] or ["- All 24 runs and DONE flags are present."])
    (out_dir / "RESULTS_GENERATED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics, configs, routing = load_results(args.results_dir)
    notes = validate_design(configs)
    final, final_notes = compute_final_losses(metrics, configs)
    notes.extend(final_notes)
    if len(final) != len(BACKBONES) * len(ATTENTION_MODES) * len(SEEDS):
        details = "; ".join(final_notes) or (
            f"expected {len(BACKBONES) * len(ATTENTION_MODES) * len(SEEDS)} "
            f"final values, found {len(final)}"
        )
        raise ValueError(
            "Final metrics are incomplete: " + details
        )
    summary = summarize(final)
    effects = compute_factorial_effects(final)
    backbone_effects = compute_backbone_effects(final)
    simple_contrasts = compute_simple_contrasts(final)
    architecture = architecture_table(configs)

    final.to_csv(args.out_dir / "final_val_loss_by_run.csv", index=False)
    summary.to_csv(args.out_dir / "condition_summary.csv", index=False)
    effects.to_csv(args.out_dir / "paired_effects_2x2x2.csv", index=False)
    backbone_effects.to_csv(args.out_dir / "paired_effects_by_backbone.csv", index=False)
    simple_contrasts.to_csv(args.out_dir / "simple_contrasts.csv", index=False)
    architecture.to_csv(args.out_dir / "architecture_summary.csv", index=False)
    if not routing.empty:
        final_iter = routing.groupby(["backbone", "attn_mode", "seed"]).iter.transform("max")
        routing[routing.iter == final_iter].to_csv(
            args.out_dir / "final_router_loads.csv", index=False
        )
    throughput_columns = [
        column for column in ("backbone", "attn_mode", "seed", "tokens_per_second", "peak_vram_gib")
        if column in metrics.columns
    ]
    metrics.dropna(subset=["val_loss"])[throughput_columns].groupby(
        ["backbone", "attn_mode", "seed"], as_index=False
    ).tail(1).to_csv(args.out_dir / "throughput_by_run.csv", index=False)

    plot_learning_curves(metrics, args.out_dir)
    plot_heatmaps(summary, args.out_dir)
    plot_final_losses(final, summary, args.out_dir)
    write_report(
        args.out_dir,
        summary,
        effects,
        backbone_effects,
        simple_contrasts,
        architecture,
        notes,
    )
    print(summary.to_string(index=False))
    print("\nSeed-paired 2x2x2 effects")
    print(effects.to_string(index=False))
    if notes:
        print("\nValidation notes:")
        for note in notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
