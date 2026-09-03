"""Analysis for the MoE-backed replication of the original 2x2 study."""

import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from model import ATTENTION_MODES, GPT, GPTConfig


CONDITIONS = {
    "mha": (False, False, "MHA"),
    "mha_rope": (False, True, "MHA + Decoupled RoPE"),
    "mla_norope": (True, False, "Low-Rank, full RoPE"),
    "mla": (True, True, "Full MLA"),
}
COLORS = {
    "mha": "#4c72b0",
    "mha_rope": "#55a868",
    "mla_norope": "#c44e52",
    "mla": "#dd8452",
}
FINAL_N_POINTS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--out_dir", type=Path, default=Path("plots"))
    parser.add_argument(
        "--dense_results_dir", type=Path, default=Path("../results")
    )
    return parser.parse_args()


def load_results(
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_frames, routing_frames, configs = [], [], []
    for run_dir in sorted(results_dir.iterdir() if results_dir.exists() else []):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        config_path, metrics_path = run_dir / "config.json", run_dir / "metrics.csv"
        if not config_path.exists() or not metrics_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        mode, seed = config.get("attn_mode"), int(config.get("seed", -1))
        if mode not in ATTENTION_MODES:
            warnings.warn(f"Ignoring unexpected mode {mode!r} in {run_dir}")
            continue
        frame = pd.read_csv(metrics_path).drop_duplicates("iter", keep="last")
        frame["attn_mode"], frame["seed"] = mode, seed
        metrics_frames.append(frame)
        routing_path = run_dir / "routing.csv"
        if routing_path.exists():
            routing = pd.read_csv(routing_path).drop_duplicates(
                ["iter", "layer", "expert"], keep="last"
            )
            routing["attn_mode"], routing["seed"] = mode, seed
            routing_frames.append(routing)
        config.update(
            {"run_dir": str(run_dir), "done": (run_dir / "DONE").exists()}
        )
        configs.append(config)
    if not metrics_frames:
        raise FileNotFoundError(f"No valid metrics found below {results_dir}")
    routing = (
        pd.concat(routing_frames, ignore_index=True)
        if routing_frames
        else pd.DataFrame()
    )
    return (
        pd.concat(metrics_frames, ignore_index=True),
        pd.DataFrame(configs),
        routing,
    )


def validate_design(configs: pd.DataFrame) -> list[str]:
    notes = []
    duplicates = configs.duplicated(["attn_mode", "seed"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate cells:\n"
            + configs.loc[duplicates, ["attn_mode", "seed", "run_dir"]].to_string()
        )
    expected = {(mode, seed) for mode in ATTENTION_MODES for seed in (42, 123, 456)}
    observed = set(zip(configs.attn_mode, configs.seed.astype(int)))
    if missing := sorted(expected - observed):
        notes.append(f"Incomplete study; missing runs: {missing}")
    controlled = [
        "n_layer",
        "n_head",
        "n_embd",
        "block_size",
        "mla_d_c",
        "mla_d_c_q",
        "mla_d_rope",
        "first_dense_layers",
        "n_shared_experts",
        "n_routed_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "aux_loss_alpha",
        "max_iters",
        "batch_size",
        "grad_accum",
        "lr",
        "min_lr",
        "warmup_iters",
        "weight_decay",
        "eval_interval",
        "eval_iters",
        "dtype",
        "compile",
    ]
    for field in controlled:
        if field in configs and configs[field].nunique(dropna=False) != 1:
            raise ValueError(f"Controlled field differs: {field}")
    for seed, group in configs.groupby("seed"):
        for field in ("train_data_seed", "val_data_seed"):
            if group[field].nunique() != 1:
                raise ValueError(f"{field} differs across modes for seed {seed}")
    incomplete = configs.loc[~configs.done, "run_dir"].tolist()
    if incomplete:
        notes.append(f"Runs without DONE flag: {incomplete}")
    return notes


def add_factor_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["low_rank"] = frame.attn_mode.map(
        {mode: values[0] for mode, values in CONDITIONS.items()}
    )
    frame["decoupled_rope"] = frame.attn_mode.map(
        {mode: values[1] for mode, values in CONDITIONS.items()}
    )
    frame["label"] = frame.attn_mode.map(
        {mode: values[2] for mode, values in CONDITIONS.items()}
    )
    return frame


def compute_final_losses(
    metrics: pd.DataFrame, configs: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    rows, notes = [], []
    lookup = {
        (row.attn_mode, int(row.seed)): row
        for row in configs.itertuples(index=False)
    }
    for (mode, seed), group in metrics.groupby(["attn_mode", "seed"]):
        config = lookup[(mode, int(seed))]
        expected = (int(config.max_iters) // int(config.eval_interval)) * int(
            config.eval_interval
        )
        valid = group.dropna(subset=["val_loss"]).sort_values("iter")
        if valid.empty or int(valid.iter.max()) < expected:
            notes.append(f"Incomplete metrics: {mode} seed={seed}")
            continue
        final_points = valid.tail(FINAL_N_POINTS)
        if len(final_points) < FINAL_N_POINTS:
            notes.append(f"Fewer than {FINAL_N_POINTS} final points: {mode} seed={seed}")
            continue
        rows.append(
            {
                "attn_mode": mode,
                "seed": int(seed),
                "final_val_loss": final_points.val_loss.mean(),
                "last_eval_iter": int(valid.iter.max()),
            }
        )
    return add_factor_metadata(pd.DataFrame(rows)), notes


def summarize(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in ATTENTION_MODES:
        values = final.loc[final.attn_mode == mode, "final_val_loss"]
        n = len(values)
        mean = values.mean() if n else float("nan")
        std = values.std(ddof=1) if n > 1 else float("nan")
        ci = stats.t.ppf(0.975, n - 1) * std / math.sqrt(n) if n > 1 else float("nan")
        rows.append(
            {
                "attn_mode": mode,
                "label": CONDITIONS[mode][2],
                "n": n,
                "mean": mean,
                "std": std,
                "ci95": ci,
                "perplexity": math.exp(mean) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def compute_paired_effects(final: pd.DataFrame) -> pd.DataFrame:
    """Test the three 2x2 effects as seed-paired one-sample contrasts.

    Each seed uses identical training and validation windows in all four cells.
    The seed is therefore the repeated-measures unit. With two levels per factor,
    the corresponding repeated-measures F test is exactly t**2 for the seed-wise
    contrast, with (1, n_seeds - 1) degrees of freedom.
    """
    pivot = final.pivot(index="seed", columns="attn_mode", values="final_val_loss")
    required = set(ATTENTION_MODES)
    if not required.issubset(pivot.columns):
        return pd.DataFrame()

    contrasts = {
        "Low-Rank": (
            (pivot["mla_norope"] + pivot["mla"]) / 2
            - (pivot["mha"] + pivot["mha_rope"]) / 2,
            "mean(mla_norope, mla) - mean(mha, mha_rope)",
        ),
        "Decoupled RoPE": (
            (pivot["mha_rope"] + pivot["mla"]) / 2
            - (pivot["mha"] + pivot["mla_norope"]) / 2,
            "mean(mha_rope, mla) - mean(mha, mla_norope)",
        ),
        "Low-Rank x Decoupled RoPE": (
            (pivot["mla"] - pivot["mla_norope"])
            - (pivot["mha_rope"] - pivot["mha"]),
            "(mla - mla_norope) - (mha_rope - mha)",
        ),
    }

    rows = []
    for effect, (values, definition) in contrasts.items():
        values = values.dropna()
        n = len(values)
        mean = values.mean() if n else float("nan")
        std = values.std(ddof=1) if n > 1 else float("nan")
        df_error = n - 1
        if n > 1:
            test = stats.ttest_1samp(values, popmean=0.0)
            t_statistic = float(test.statistic)
            p_value = float(test.pvalue)
            ci_half_width = stats.t.ppf(0.975, df_error) * std / math.sqrt(n)
            f_statistic = t_statistic**2
            partial_eta_sq = f_statistic / (f_statistic + df_error)
        else:
            t_statistic = p_value = ci_half_width = float("nan")
            f_statistic = partial_eta_sq = float("nan")
        rows.append(
            {
                "effect": effect,
                "definition": definition,
                "n_pairs": n,
                "mean_difference": mean,
                "std_difference": std,
                "ci95_lower": mean - ci_half_width,
                "ci95_upper": mean + ci_half_width,
                "t_statistic": t_statistic,
                "df_error": df_error,
                "f_statistic": f_statistic,
                "p_value": p_value,
                "partial_eta_sq": partial_eta_sq,
            }
        )
    return pd.DataFrame(rows)


def architecture_table(configs: pd.DataFrame) -> pd.DataFrame:
    first = configs.iloc[0]
    rows = []
    for mode in ATTENTION_MODES:
        config = GPTConfig(
            block_size=int(first.block_size),
            n_layer=int(first.n_layer),
            n_head=int(first.n_head),
            n_embd=int(first.n_embd),
            attn_mode=mode,
            mla_d_c=int(first.mla_d_c),
            mla_d_c_q=int(first.mla_d_c_q),
            mla_d_rope=int(first.mla_d_rope),
            first_dense_layers=int(first.first_dense_layers),
            n_shared_experts=int(first.n_shared_experts),
            n_routed_experts=int(first.n_routed_experts),
            num_experts_per_tok=int(first.num_experts_per_tok),
            moe_intermediate_size=int(first.moe_intermediate_size),
            aux_loss_alpha=float(first.aux_loss_alpha),
        )
        model = GPT(config)
        rows.append(
            {
                "attn_mode": mode,
                "total_parameters": model.count_parameters(),
                "active_parameters_per_token": model.count_active_parameters(),
            }
        )
    return pd.DataFrame(rows)


def plot_learning_curves(metrics: pd.DataFrame, out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, column, title in zip(
        axes, ("train_loss", "val_loss"), ("Training CE", "Validation CE")
    ):
        for mode in ATTENTION_MODES:
            values = metrics[metrics.attn_mode == mode].groupby("iter")[column].agg(
                ["mean", "std"]
            )
            x = values.index.to_numpy(float)
            mean = values["mean"].to_numpy(float)
            std = values["std"].fillna(0).to_numpy(float)
            axis.plot(x, mean, color=COLORS[mode], label=CONDITIONS[mode][2])
            axis.fill_between(x, mean - std, mean + std, color=COLORS[mode], alpha=0.15)
        axis.set(title=title, xlabel="Iteration", ylabel="Cross-entropy loss")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("MoE backbone: original 2x2 attention ablation")
    figure.tight_layout()
    figure.savefig(out_dir / "learning_curves.png", dpi=170)
    plt.close(figure)


def plot_final_losses(final: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    x = np.arange(4)
    means = [summary.loc[summary.attn_mode == mode, "mean"].iloc[0] for mode in ATTENTION_MODES]
    cis = [summary.loc[summary.attn_mode == mode, "ci95"].iloc[0] for mode in ATTENTION_MODES]
    axis.bar(x, means, yerr=cis, capsize=6, color=[COLORS[m] for m in ATTENTION_MODES], alpha=0.8)
    for seed, row in final.pivot(index="seed", columns="attn_mode", values="final_val_loss").iterrows():
        values = [row.get(mode, np.nan) for mode in ATTENTION_MODES]
        axis.plot(x, values, color="black", alpha=0.3)
        axis.scatter(x, values, color="black", s=26)
    axis.set_xticks(x, [CONDITIONS[m][2] for m in ATTENTION_MODES], rotation=8)
    axis.set_ylabel("Final validation CE (mean of last 5 evals)")
    axis.set_title("MoE 2x2: final validation loss")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_dir / "final_val_loss.png", dpi=170)
    plt.close(figure)


def plot_heatmap(summary: pd.DataFrame, out_dir: Path) -> None:
    lookup = summary.set_index("attn_mode")["mean"]
    matrix = np.array(
        [[lookup.get("mha", np.nan), lookup.get("mha_rope", np.nan)],
         [lookup.get("mla_norope", np.nan), lookup.get("mla", np.nan)]]
    )
    figure, axis = plt.subplots(figsize=(7, 5.5))
    image = axis.imshow(matrix, cmap="RdYlGn_r")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row, column]:.4f}", ha="center", va="center", fontsize=13)
    axis.set_xticks([0, 1], ["No", "Yes"])
    axis.set_yticks([0, 1], ["No", "Yes"])
    axis.set_xlabel("Decoupled RoPE")
    axis.set_ylabel("Low-Rank KV/Q")
    axis.set_title("MoE backbone: 2x2 cell means")
    figure.colorbar(image, ax=axis, label="Final validation loss")
    figure.tight_layout()
    figure.savefig(out_dir / "heatmap_2x2.png", dpi=170)
    plt.close(figure)


def plot_router_diagnostics(routing: pd.DataFrame, out_dir: Path) -> None:
    if routing.empty:
        return
    final_iteration = routing.groupby(["attn_mode", "seed"]).iter.transform("max")
    final = routing[routing.iter == final_iteration]
    load = final.groupby(["layer", "expert"]).selected_fraction.mean().unstack()
    figure, axis = plt.subplots(figsize=(12, 4.5))
    image = axis.imshow(load.to_numpy(), aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(load.columns)), load.columns)
    axis.set_yticks(range(len(load.index)), load.index)
    axis.set_xlabel("Routed expert")
    axis.set_ylabel("MoE layer")
    axis.set_title("Final routed-expert selection fractions (all cells/seeds)")
    figure.colorbar(image, ax=axis, label="Fraction of top-k assignments")
    figure.tight_layout()
    figure.savefig(out_dir / "router_load_heatmap.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for mode in ATTENTION_MODES:
        subset = routing[routing.attn_mode == mode]
        entropy = subset.groupby(["iter", "seed", "layer"]).router_entropy.first().groupby("iter").mean()
        maximum = subset.groupby(["iter", "seed", "layer"]).selected_fraction.max().groupby("iter").mean()
        axes[0].plot(entropy.index, entropy.values, color=COLORS[mode], label=CONDITIONS[mode][2])
        axes[1].plot(maximum.index, maximum.values, color=COLORS[mode], label=CONDITIONS[mode][2])
    axes[0].set(title="Router entropy", xlabel="Iteration", ylabel="Mean entropy")
    axes[1].set(title="Maximum expert load", xlabel="Iteration", ylabel="Selection fraction")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out_dir / "router_balance.png", dpi=170)
    plt.close(figure)


def load_dense_final(path: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in path.glob("*/metrics.csv") if path.exists() else []:
        if metrics_path.parent.name.startswith("_"):
            continue
        frame = pd.read_csv(metrics_path)
        if "attn_mode" not in frame or "seed" not in frame:
            continue
        frame = frame.dropna(subset=["val_loss"])
        for (mode, seed), group in frame.groupby(["attn_mode", "seed"]):
            if mode in ATTENTION_MODES and len(group) >= FINAL_N_POINTS:
                rows.append(
                    {"attn_mode": mode, "seed": int(seed), "final_val_loss": group.nlargest(FINAL_N_POINTS, "iter").val_loss.mean()}
                )
    return pd.DataFrame(rows)


def plot_dense_comparison(
    dense: pd.DataFrame, moe: pd.DataFrame, out_dir: Path
) -> pd.DataFrame:
    if dense.empty or moe.empty:
        return pd.DataFrame()
    dense_summary = dense.groupby("attn_mode").final_val_loss.agg(dense_mean="mean", dense_std="std")
    moe_summary = moe.groupby("attn_mode").final_val_loss.agg(moe_mean="mean", moe_std="std")
    comparison = dense_summary.join(moe_summary).reset_index()
    comparison["moe_minus_dense"] = comparison.moe_mean - comparison.dense_mean
    figure, axis = plt.subplots(figsize=(10, 5.5))
    x = np.arange(4)
    width = 0.36
    indexed = comparison.set_index("attn_mode")
    axis.bar(x - width / 2, [indexed.loc[m, "dense_mean"] for m in ATTENTION_MODES], width, label="Original dense")
    axis.bar(x + width / 2, [indexed.loc[m, "moe_mean"] for m in ATTENTION_MODES], width, label="MoE follow-up")
    axis.set_xticks(x, [CONDITIONS[m][2] for m in ATTENTION_MODES], rotation=8)
    axis.set_ylabel("Final validation loss")
    axis.set_title("Descriptive cross-backbone comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_dir / "dense_vs_moe.png", dpi=170)
    plt.close(figure)
    return comparison


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    notes: list[str],
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "(no rows)"
        columns = [str(column) for column in frame.columns]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in frame.itertuples(index=False, name=None):
            cells = []
            for value in row:
                if isinstance(value, (float, np.floating)):
                    cells.append(f"{value:.6g}")
                else:
                    cells.append(str(value))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    lines = [
        "# Generated results: MoE 2x2 follow-up",
        "",
        "Final validation loss is cross-entropy only; the router auxiliary loss is not included.",
        "",
        "## Condition summary",
        "",
        markdown_table(summary),
        "",
        "## Seed-paired 2x2 effects",
        "",
        "Each effect is a seed-wise contrast tested against zero. The repeated-measures "
        "F statistic is t^2 with df=(1, n_pairs-1).",
        "",
        markdown_table(effects) if not effects.empty else "Study incomplete.",
        "",
        "## Validation notes",
        "",
    ]
    lines.extend([f"- {note}" for note in notes] or ["- Complete design and DONE flags detected."])
    (out_dir / "RESULTS_GENERATED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics, configs, routing = load_results(args.results_dir)
    notes = validate_design(configs)
    final, final_notes = compute_final_losses(metrics, configs)
    notes.extend(final_notes)
    summary = summarize(final)
    effects = compute_paired_effects(final)
    architecture = architecture_table(configs)
    plot_learning_curves(metrics, args.out_dir)
    plot_final_losses(final, summary, args.out_dir)
    plot_heatmap(summary, args.out_dir)
    plot_router_diagnostics(routing, args.out_dir)
    dense = load_dense_final(args.dense_results_dir)
    comparison = plot_dense_comparison(dense, final, args.out_dir)
    final.to_csv(args.out_dir / "final_val_loss_by_run.csv", index=False)
    summary.to_csv(args.out_dir / "condition_summary.csv", index=False)
    effects.to_csv(args.out_dir / "anova_2x2.csv", index=False)
    architecture.to_csv(args.out_dir / "architecture_summary.csv", index=False)
    if not comparison.empty:
        comparison.to_csv(args.out_dir / "dense_vs_moe_summary.csv", index=False)
    if not routing.empty:
        final_iteration = routing.groupby(["attn_mode", "seed"]).iter.transform("max")
        routing[routing.iter == final_iteration].to_csv(
            args.out_dir / "final_router_loads.csv", index=False
        )
    write_report(args.out_dir, summary, effects, notes)
    print(summary.to_string(index=False))
    print("\nSeed-paired 2x2 effects")
    print(effects.to_string(index=False))
    if notes:
        print("\nValidation notes:")
        for note in notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
