"""
Analysis script for the 2×2 ablation study results.

Reads all metrics.csv files from the results/ directory and produces:
  1. Learning curves (train + val loss per condition, mean ± std over seeds)
  2. Final validation loss bar chart with 95% confidence intervals
  3. 2-way ANOVA table (Low-Rank × Decoupled-RoPE, with interaction)
  4. Partial η² effect sizes
  5. Summary table printed to stdout

Usage:
  cd experiments
  python analyze_results.py [--results_dir results] [--out_dir plots]

Expected results/ layout:
  results/
    mha_s42/metrics.csv
    mha_s123/metrics.csv
    mha_s456/metrics.csv
    mha_rope_s42/metrics.csv
    ...  (12 CSV files total)
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not installed — ANOVA will be skipped")


# ---------------------------------------------------------------------------
# Condition metadata
# ---------------------------------------------------------------------------

CONDITIONS = {
    "mha":        {"low_rank": False, "decoupled_rope": False, "label": "MHA (Baseline)"},
    "mha_rope":   {"low_rank": False, "decoupled_rope": True,  "label": "MHA + Decoupled RoPE"},
    "mla_norope": {"low_rank": True,  "decoupled_rope": False, "label": "MLA w/o Decoupled RoPE"},
    "mla":        {"low_rank": True,  "decoupled_rope": True,  "label": "Full MLA"},
}

COLORS = {
    "mha":        "#4c72b0",
    "mha_rope":   "#55a868",
    "mla_norope": "#c44e52",
    "mla":        "#dd8452",
}

# How many final eval points to average for "final val loss"
FINAL_N_POINTS = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> pd.DataFrame:
    """
    Load all metrics.csv files and return a combined DataFrame with columns:
      iter, train_loss, val_loss, lr, tokens_seen, elapsed_s,
      attn_mode, seed, low_rank, decoupled_rope, condition_label
    """
    dfs = []
    for csv_path in sorted(results_dir.glob("*/metrics.csv")):
        # Skip non-experiment dirs like _smoke_test/ — they share attn_mode/seed
        # values with real runs and would contaminate the learning curves.
        if csv_path.parent.name.startswith("_"):
            continue
        df = pd.read_csv(csv_path)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No metrics.csv files found in {results_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    combined["attn_mode"] = combined["attn_mode"].str.strip()
    # A crash between eval and checkpoint can re-log iters after resume;
    # keep the last row per (attn_mode, seed, iter).
    combined = combined.drop_duplicates(subset=["attn_mode", "seed", "iter"], keep="last")

    for mode, meta in CONDITIONS.items():
        mask = combined["attn_mode"] == mode
        combined.loc[mask, "low_rank"]       = meta["low_rank"]
        combined.loc[mask, "decoupled_rope"] = meta["decoupled_rope"]
        combined.loc[mask, "condition_label"] = meta["label"]

    combined["low_rank"]       = combined["low_rank"].astype(bool)
    combined["decoupled_rope"] = combined["decoupled_rope"].astype(bool)
    return combined


def compute_final_val_loss(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (attn_mode, seed), compute final val loss as mean over last FINAL_N_POINTS evals.
    Returns a DataFrame with one row per run.
    """
    rows = []
    for (mode, seed), grp in df.dropna(subset=["val_loss"]).groupby(["attn_mode", "seed"]):
        last_n = grp.nlargest(FINAL_N_POINTS, "iter")["val_loss"]
        final_loss = last_n.mean()
        rows.append({
            "attn_mode":       mode,
            "seed":            int(seed),
            "final_val_loss":  final_loss,
            "low_rank":        CONDITIONS[mode]["low_rank"],
            "decoupled_rope":  CONDITIONS[mode]["decoupled_rope"],
            "label":           CONDITIONS[mode]["label"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_learning_curves(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric in zip(axes, ["train_loss", "val_loss"]):
        for mode, meta in CONDITIONS.items():
            grp = df[(df["attn_mode"] == mode) & df[metric].notna()]
            if grp.empty:
                continue
            pivot = grp.groupby(["iter", "seed"])[metric].mean().unstack("seed")
            iters = pivot.index.values
            mean  = pivot.mean(axis=1).values
            std   = pivot.std(axis=1).fillna(0).values
            color = COLORS[mode]
            ax.plot(iters, mean, label=meta["label"], color=color, linewidth=1.8)
            ax.fill_between(iters, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_xlabel("Training iteration")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title("Training Loss" if metric == "train_loss" else "Validation Loss")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "learning_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_final_val_loss(final_df: pd.DataFrame, out_dir: Path) -> None:
    order = ["mha", "mha_rope", "mla_norope", "mla"]
    labels = [CONDITIONS[m]["label"] for m in order]
    colors = [COLORS[m] for m in order]

    means, cis = [], []
    for mode in order:
        vals = final_df[final_df["attn_mode"] == mode]["final_val_loss"]
        n = len(vals)
        means.append(vals.mean())
        # 95% CI: t_{n-1, 0.975} * SEM
        t_crit = stats.t.ppf(0.975, df=max(n - 1, 1))
        cis.append(t_crit * vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(order))
    bars = ax.bar(x, means, yerr=cis, capsize=5, color=colors, alpha=0.85,
                  error_kw={"elinewidth": 1.5, "capthick": 1.5})

    # Individual seed points
    for i, mode in enumerate(order):
        vals = final_df[final_df["attn_mode"] == mode]["final_val_loss"].values
        ax.scatter(np.full_like(vals, i), vals, color="black", s=25, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Final Validation Loss\n(mean of last 5 eval points, ±95% CI)")
    ax.set_title("2×2 Ablation: Final Validation Loss by Condition")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=min(means) * 0.98)

    plt.tight_layout()
    path = out_dir / "final_val_loss.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_2x2_heatmap(final_df: pd.DataFrame, out_dir: Path) -> None:
    """2×2 grid showing mean final val loss per cell."""
    fig, ax = plt.subplots(figsize=(6, 5))

    matrix = np.zeros((2, 2))
    labels_mat = [["", ""], ["", ""]]
    for mode, meta in CONDITIONS.items():
        row = 1 if meta["low_rank"] else 0
        col = 1 if meta["decoupled_rope"] else 0
        vals = final_df[final_df["attn_mode"] == mode]["final_val_loss"]
        mean = vals.mean()
        matrix[row, col] = mean
        labels_mat[row][col] = f"{meta['label']}\n{mean:.3f}"

    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Final Val Loss")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Decoupled RoPE", "Decoupled RoPE"])
    ax.set_yticklabels(["No Low-Rank", "Low-Rank KV"])
    ax.set_xlabel("Decoupled RoPE factor")
    ax.set_ylabel("Low-Rank KV factor")
    ax.set_title("2×2 Ablation Heatmap")

    for (r, c), lab in [((r, c), labels_mat[r][c]) for r in range(2) for c in range(2)]:
        ax.text(c, r, lab, ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if matrix[r, c] > matrix.mean() else "black")

    plt.tight_layout()
    path = out_dir / "heatmap_2x2.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def run_anova(final_df: pd.DataFrame) -> None:
    """Two-way ANOVA with interaction: final_val_loss ~ low_rank * decoupled_rope."""
    if not HAS_STATSMODELS:
        print("\n[ANOVA skipped — statsmodels not installed]")
        return

    df = final_df.copy()
    df["low_rank_i"]       = df["low_rank"].astype(int)
    df["decoupled_rope_i"] = df["decoupled_rope"].astype(int)

    formula = "final_val_loss ~ C(low_rank_i) * C(decoupled_rope_i)"
    model   = smf.ols(formula, data=df).fit()
    table   = anova_lm(model, typ=2)

    print("\n--- 2-Way ANOVA (Type II SS) ---")
    print(table.to_string())

    # Partial η² = SS_effect / (SS_effect + SS_residual)
    ss_res = table.loc["Residual", "sum_sq"]
    print("\n--- Partial η² ---")
    for term in table.index:
        if term == "Residual":
            continue
        ss = table.loc[term, "sum_sq"]
        eta2 = ss / (ss + ss_res)
        print(f"  {term:40s}: partial η² = {eta2:.4f}")


def print_summary(final_df: pd.DataFrame) -> None:
    print("\n--- Final Validation Loss Summary ---")
    print(f"{'Condition':<30} {'n':>4}  {'mean':>8}  {'std':>8}  {'95% CI':>14}")
    for mode in ("mha", "mha_rope", "mla_norope", "mla"):
        vals = final_df[final_df["attn_mode"] == mode]["final_val_loss"]
        n = len(vals)
        mean = vals.mean()
        std  = vals.std(ddof=1) if n > 1 else float("nan")
        t_crit = stats.t.ppf(0.975, df=max(n - 1, 1)) if n > 1 else float("nan")
        ci = t_crit * std / np.sqrt(n) if n > 1 else float("nan")
        label = CONDITIONS[mode]["label"]
        print(f"  {label:<28} {n:>4}  {mean:>8.4f}  {std:>8.4f}  ±{ci:>6.4f}")

    print("\n--- Main Effects (cell means) ---")
    # Low-rank main effect
    lr_no  = final_df[~final_df["low_rank"]]["final_val_loss"].mean()
    lr_yes = final_df[ final_df["low_rank"]]["final_val_loss"].mean()
    print(f"  Low-Rank effect:         {lr_no:.4f} (no) vs {lr_yes:.4f} (yes) → Δ = {lr_yes - lr_no:+.4f}")
    # Decoupled RoPE main effect
    dr_no  = final_df[~final_df["decoupled_rope"]]["final_val_loss"].mean()
    dr_yes = final_df[ final_df["decoupled_rope"]]["final_val_loss"].mean()
    print(f"  Decoupled-RoPE effect:   {dr_no:.4f} (no) vs {dr_yes:.4f} (yes) → Δ = {dr_yes - dr_no:+.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out_dir",     default="plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    df = load_results(results_dir)
    print(f"Loaded {len(df):,} rows from {df['attn_mode'].nunique()} conditions, "
          f"{df['seed'].nunique()} seeds")

    final_df = compute_final_val_loss(df)
    print(f"Final-loss table: {len(final_df)} runs")

    print_summary(final_df)
    run_anova(final_df)

    plot_learning_curves(df, out_dir)
    plot_final_val_loss(final_df, out_dir)
    plot_2x2_heatmap(final_df, out_dir)

    # Save final loss table
    csv_path = out_dir / "final_val_loss_summary.csv"
    final_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
