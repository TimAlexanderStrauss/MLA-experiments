"""Analyze the clean paired DeepSeek-layout sensitivity study.

Outputs:
  plots/learning_curves.png
  plots/final_val_loss.png
  plots/paired_differences.png
  plots/quality_vs_cache.png
  plots/final_val_loss_by_run.csv
  plots/condition_summary.csv
  plots/paired_contrasts.csv
  plots/RESULTS_GENERATED.md
"""

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

from model import GPT, GPTConfig


MODES = ("mha", "mla_current", "mla_deepseek")
LABELS = {
    "mha": "MHA",
    "mla_current": "MLA (original split layout)",
    "mla_deepseek": "MLA (DeepSeek-like layout)",
}
COLORS = {
    "mha": "#4c72b0",
    "mla_current": "#dd8452",
    "mla_deepseek": "#55a868",
}
FINAL_N_POINTS = 5


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    configs = []
    for run_dir in sorted(results_dir.iterdir() if results_dir.exists() else []):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.csv"
        if not config_path.exists() or not metrics_path.exists():
            continue

        config = json.loads(config_path.read_text(encoding="utf-8"))
        mode = config.get("attn_mode")
        seed = int(config.get("seed", -1))
        if mode not in MODES:
            warnings.warn(f"Skipping unexpected mode {mode!r} in {run_dir}")
            continue

        frame = pd.read_csv(metrics_path)
        frame = frame.drop_duplicates(subset=["iter"], keep="last")
        frame["attn_mode"] = mode
        frame["seed"] = seed
        frame["run_dir"] = str(run_dir)
        frames.append(frame)

        config_row = dict(config)
        config_row["run_dir"] = str(run_dir)
        config_row["done"] = (run_dir / "DONE").exists()
        configs.append(config_row)

    if not frames:
        raise FileNotFoundError(f"No valid runs found in {results_dir}")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(configs)


def validate_design(configs: pd.DataFrame) -> list[str]:
    notes = []
    duplicate = configs.duplicated(subset=["attn_mode", "seed"], keep=False)
    if duplicate.any():
        duplicated = configs.loc[duplicate, ["attn_mode", "seed", "run_dir"]]
        raise ValueError(f"Duplicate mode/seed runs:\n{duplicated}")

    expected = {(mode, seed) for mode in MODES for seed in (42, 123, 456)}
    observed = set(zip(configs["attn_mode"], configs["seed"].astype(int)))
    missing = sorted(expected - observed)
    if missing:
        notes.append(f"Incomplete study; missing runs: {missing}")

    controlled_fields = [
        "n_layer",
        "n_head",
        "n_embd",
        "block_size",
        "mla_d_c",
        "mla_d_c_q",
        "mla_d_rope",
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
    for field in controlled_fields:
        if field in configs and configs[field].nunique(dropna=False) != 1:
            raise ValueError(
                f"Controlled field {field!r} differs across runs: "
                f"{configs[field].unique().tolist()}"
            )

    # The main fairness invariant: each seed has identical data RNG seeds in
    # all model conditions.
    for seed, group in configs.groupby("seed"):
        for field in ("train_data_seed", "val_data_seed"):
            if group[field].nunique() != 1:
                raise ValueError(
                    f"{field} differs across modes for experimental seed {seed}"
                )

    incomplete_flags = configs.loc[~configs["done"], "run_dir"].tolist()
    if incomplete_flags:
        notes.append(f"Runs without DONE flag: {incomplete_flags}")
    return notes


def compute_final_losses(
    metrics: pd.DataFrame, configs: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    notes = []
    config_lookup = {
        (row.attn_mode, int(row.seed)): row
        for row in configs.itertuples(index=False)
    }
    for (mode, seed), group in metrics.groupby(["attn_mode", "seed"]):
        config = config_lookup[(mode, int(seed))]
        expected_last_eval = (
            int(config.max_iters) // int(config.eval_interval)
        ) * int(config.eval_interval)
        valid = group.dropna(subset=["val_loss"]).sort_values("iter")
        actual_last_eval = int(valid["iter"].max())
        if actual_last_eval < expected_last_eval:
            notes.append(
                f"Incomplete metrics for {mode} seed={seed}: "
                f"last eval {actual_last_eval}, expected {expected_last_eval}"
            )
            continue
        final_points = valid.tail(FINAL_N_POINTS)
        if len(final_points) < FINAL_N_POINTS:
            notes.append(f"Too few final points for {mode} seed={seed}")
            continue
        rows.append(
            {
                "attn_mode": mode,
                "label": LABELS[mode],
                "seed": int(seed),
                "final_val_loss": final_points["val_loss"].mean(),
                "last_eval_iter": actual_last_eval,
                "train_data_seed": int(config.train_data_seed),
                "val_data_seed": int(config.val_data_seed),
            }
        )
    return pd.DataFrame(rows), notes


def summarize_conditions(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in MODES:
        values = final.loc[final["attn_mode"] == mode, "final_val_loss"]
        n = len(values)
        mean = values.mean() if n else float("nan")
        std = values.std(ddof=1) if n > 1 else float("nan")
        ci = (
            stats.t.ppf(0.975, n - 1) * std / math.sqrt(n)
            if n > 1
            else float("nan")
        )
        rows.append(
            {
                "attn_mode": mode,
                "label": LABELS[mode],
                "n": n,
                "mean": mean,
                "std": std,
                "ci95": ci,
                "perplexity": math.exp(mean) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def paired_contrast(
    final: pd.DataFrame,
    left: str,
    right: str,
    name: str,
    alternative: str,
) -> tuple[dict, pd.DataFrame]:
    pivot = final.pivot(index="seed", columns="attn_mode", values="final_val_loss")
    paired = pivot.reindex(columns=[left, right]).dropna().copy()
    paired["difference"] = paired[left] - paired[right]
    differences = paired["difference"]
    n = len(differences)
    mean = differences.mean() if n else float("nan")
    std = differences.std(ddof=1) if n > 1 else float("nan")
    ci = (
        stats.t.ppf(0.975, n - 1) * std / math.sqrt(n)
        if n > 1
        else float("nan")
    )
    if n > 1:
        test = stats.ttest_rel(
            paired[left], paired[right], alternative=alternative
        )
        statistic, p_value = float(test.statistic), float(test.pvalue)
    else:
        statistic, p_value = float("nan"), float("nan")
    row = {
        "contrast": name,
        "left_mode": left,
        "right_mode": right,
        "definition": f"{left} - {right}",
        "n_pairs": n,
        "mean_difference": mean,
        "std_difference": std,
        "ci95_half_width": ci,
        "alternative": alternative,
        "t_statistic": statistic,
        "p_value": p_value,
    }
    paired = paired.reset_index()
    paired["contrast"] = name
    return row, paired


def compute_contrasts(final: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    specifications = [
        (
            "mla_deepseek",
            "mla_current",
            "PRIMARY: DeepSeek-like vs current MLA",
            "less",
        ),
        (
            "mla_deepseek",
            "mha",
            "SECONDARY: DeepSeek-like MLA vs MHA",
            "two-sided",
        ),
        (
            "mla_current",
            "mha",
            "CONTEXT: current MLA vs MHA",
            "two-sided",
        ),
    ]
    rows, details = [], []
    for specification in specifications:
        row, paired = paired_contrast(final, *specification)
        rows.append(row)
        details.append(paired)
    return pd.DataFrame(rows), pd.concat(details, ignore_index=True)


def parameter_and_cache_table(configs: pd.DataFrame) -> pd.DataFrame:
    first = configs.iloc[0]
    rows = []
    for mode in MODES:
        config = GPTConfig(
            block_size=int(first.block_size),
            vocab_size=50257,
            n_layer=int(first.n_layer),
            n_head=int(first.n_head),
            n_embd=int(first.n_embd),
            dropout=0.0,
            bias=False,
            attn_mode=mode,
            mla_d_c=int(first.mla_d_c),
            mla_d_c_q=int(first.mla_d_c_q),
            mla_d_rope=int(first.mla_d_rope),
        )
        d_h = config.n_embd // config.n_head
        if mode == "mha":
            cache = 2 * config.n_head * d_h
        elif mode == "mla_current":
            cache = config.mla_d_c + config.n_head * config.mla_d_rope
        else:
            cache = config.mla_d_c + config.mla_d_rope
        rows.append(
            {
                "attn_mode": mode,
                "parameters": GPT(config).count_parameters(),
                "cache_scalars_per_token_layer": cache,
                "cache_reduction_vs_mha": (
                    2 * config.n_head * d_h / cache
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_learning_curves(metrics: pd.DataFrame, out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    for axis, column, title in [
        (axes[0], "train_loss", "Training Loss"),
        (axes[1], "val_loss", "Validation Loss"),
    ]:
        for mode in MODES:
            subset = metrics[metrics["attn_mode"] == mode]
            aggregate = subset.groupby("iter")[column].agg(["mean", "std"])
            x = aggregate.index.to_numpy(dtype=float)
            mean = aggregate["mean"].to_numpy(dtype=float)
            std = aggregate["std"].fillna(0).to_numpy(dtype=float)
            axis.plot(x, mean, label=LABELS[mode], color=COLORS[mode], linewidth=2)
            axis.fill_between(
                x, mean - std, mean + std, color=COLORS[mode], alpha=0.16
            )
        axis.set_title(title)
        axis.set_xlabel("Training iteration")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out_dir / "learning_curves.png", dpi=160)
    plt.close(figure)


def plot_final_losses(
    final: pd.DataFrame, summary: pd.DataFrame, out_dir: Path
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    x = np.arange(len(MODES))
    means = [summary.loc[summary.attn_mode == mode, "mean"].iloc[0] for mode in MODES]
    cis = [summary.loc[summary.attn_mode == mode, "ci95"].iloc[0] for mode in MODES]
    axis.bar(
        x,
        means,
        yerr=cis,
        capsize=6,
        color=[COLORS[mode] for mode in MODES],
        alpha=0.78,
    )
    pivot = final.pivot(index="seed", columns="attn_mode", values="final_val_loss")
    for seed, row in pivot.iterrows():
        if all(mode in row and pd.notna(row[mode]) for mode in MODES):
            values = [row[mode] for mode in MODES]
            axis.plot(x, values, color="black", alpha=0.28, linewidth=1)
            axis.scatter(x, values, color="black", s=28, zorder=4)
            axis.annotate(
                str(seed), (x[-1], values[-1]), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=8
            )
    axis.set_xticks(x, [LABELS[mode] for mode in MODES], rotation=8)
    axis.set_ylabel("Final Validation Loss (last 5 evals)")
    axis.set_title("Paper-layout sensitivity: paired final losses")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_dir / "final_val_loss.png", dpi=160)
    plt.close(figure)


def plot_paired_differences(details: pd.DataFrame, out_dir: Path) -> None:
    contrasts = details["contrast"].unique().tolist()
    figure, axis = plt.subplots(figsize=(11, 5.5))
    for position, contrast in enumerate(contrasts):
        values = details.loc[details["contrast"] == contrast, "difference"]
        jitter = np.linspace(-0.06, 0.06, len(values))
        axis.scatter(
            position + jitter,
            values,
            color="#333333",
            s=42,
            alpha=0.8,
            zorder=3,
        )
        if len(values):
            axis.scatter(
                position,
                values.mean(),
                marker="D",
                color="#c44e52",
                s=75,
                zorder=4,
            )
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(range(len(contrasts)), contrasts, rotation=8)
    axis.set_ylabel("Paired loss difference (left - right)\nnegative = left is better")
    axis.set_title("Seed-paired condition differences")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_dir / "paired_differences.png", dpi=160)
    plt.close(figure)


def plot_quality_vs_cache(
    summary: pd.DataFrame, architecture: pd.DataFrame, out_dir: Path
) -> None:
    merged = architecture.merge(summary, on="attn_mode", how="left")
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for row in merged.itertuples(index=False):
        axis.errorbar(
            row.cache_scalars_per_token_layer,
            row.mean,
            yerr=row.ci95,
            fmt="o",
            color=COLORS[row.attn_mode],
            markersize=9,
            capsize=5,
            label=LABELS[row.attn_mode],
        )
        axis.annotate(
            f"{row.cache_scalars_per_token_layer} scalars\n"
            f"{row.cache_reduction_vs_mha:.2f}x reduction",
            (row.cache_scalars_per_token_layer, row.mean),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Theoretical KV-cache scalars per token and layer")
    axis.set_ylabel("Final Validation Loss")
    axis.set_title("Quality vs. theoretical KV cache (lower-left is better)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out_dir / "quality_vs_cache.png", dpi=160)
    plt.close(figure)


def fmt(value: float, digits: int = 4) -> str:
    return "n/a" if pd.isna(value) else f"{value:.{digits}f}"


def write_report(
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    architecture: pd.DataFrame,
    notes: list[str],
    out_dir: Path,
) -> None:
    merged = summary.merge(architecture, on="attn_mode")
    lines = [
        "# Automatisch erzeugte Ergebnisse: Paper-Layout-Follow-up",
        "",
        "> Diese Datei wird von `analyze_results.py` aus den Rohdaten erzeugt.",
        "",
        "## Bedingungen",
        "",
        "| Bedingung | n | Val Loss | Std | 95-%-CI | PPL | Parameter | Cache-Werte | Reduktion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in merged.itertuples(index=False):
        lines.append(
            f"| {row.label} | {row.n} | {fmt(row.mean)} | {fmt(row.std)} | "
            f"+/-{fmt(row.ci95)} | {fmt(row.perplexity, 2)} | "
            f"{row.parameters:,} | {row.cache_scalars_per_token_layer} | "
            f"{row.cache_reduction_vs_mha:.2f}x |"
        )
    lines.extend(
        [
            "",
            "## Vorab definierte gepaarte Vergleiche",
            "",
            "Eine negative Differenz bedeutet, dass die links genannte Bedingung besser ist.",
            "",
            "| Vergleich | n | Mean Diff | 95-%-CI | Alternative | t | p |",
            "|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in contrasts.itertuples(index=False):
        lines.append(
            f"| {row.contrast} | {row.n_pairs} | {fmt(row.mean_difference)} | "
            f"{fmt(row.mean_difference - row.ci95_half_width)} bis "
            f"{fmt(row.mean_difference + row.ci95_half_width)} | "
            f"{row.alternative} | {fmt(row.t_statistic, 3)} | "
            f"{fmt(row.p_value, 5)} |"
        )
    lines.extend(
        [
            "",
            "## Abbildungen",
            "",
            "![Lernkurven](learning_curves.png)",
            "",
            "![Finale Loss](final_val_loss.png)",
            "",
            "![Gepaarte Differenzen](paired_differences.png)",
            "",
            "![Qualität gegen Cache](quality_vs_cache.png)",
            "",
            "## Interpretationshinweis",
            "",
            "Mit nur drei Seed-Paaren sind Effektgrößen und die Konsistenz der "
            "Einzelseeds wichtiger als ein isolierter p-Wert. Das primäre gerichtete "
            "Ergebnis ist `mla_deepseek < mla_current`. Der Vergleich zu MHA ist "
            "sekundär und zweiseitig.",
        ]
    )
    if notes:
        lines.extend(["", "## Warnungen", ""])
        lines.extend([f"- {note}" for note in notes])
    (out_dir / "RESULTS_GENERATED.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def print_tables(summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    print("\nCondition summary")
    print(summary.to_string(index=False))
    print("\nPaired contrasts")
    print(contrasts.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out_dir", default="plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, configs = load_results(results_dir)
    notes = validate_design(configs)
    final, final_notes = compute_final_losses(metrics, configs)
    notes.extend(final_notes)
    if final.empty:
        raise RuntimeError("No complete runs with final evaluation points found")

    summary = summarize_conditions(final)
    contrasts, contrast_details = compute_contrasts(final)
    architecture = parameter_and_cache_table(configs)
    print_tables(summary, contrasts)
    for note in notes:
        print(f"WARNING: {note}")

    final.to_csv(out_dir / "final_val_loss_by_run.csv", index=False)
    summary.to_csv(out_dir / "condition_summary.csv", index=False)
    contrasts.to_csv(out_dir / "paired_contrasts.csv", index=False)
    architecture.to_csv(out_dir / "architecture_summary.csv", index=False)

    plot_learning_curves(metrics, out_dir)
    plot_final_losses(final, summary, out_dir)
    plot_paired_differences(contrast_details, out_dir)
    plot_quality_vs_cache(summary, architecture, out_dir)
    write_report(summary, contrasts, architecture, notes, out_dir)
    print(f"\nSaved analysis to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
