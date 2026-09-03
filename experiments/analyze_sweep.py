"""
analyze_sweep.py - d_c compression-strength sweep analysis for mla_norope.

Reads new sweep runs from results_sweep/ and reuses d_c=128 baseline from
results/mla_norope_s*, then produces dose-response and Pareto plots.

NOTE: stdout uses ASCII-only characters (+/-, eta2, ->) to avoid Windows
cp1252 encoding errors.

Usage:
  cd experiments
  python analyze_sweep.py [--results_dir results] [--sweep_dir results_sweep] [--out_dir plots]
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist


FINAL_N_POINTS = 5

# Theoretical KV-cache sizes per token per layer (scalars/float values). See RESULTS.md §6.
# mla_norope: cache = d_c (the compressed KV latent only)
# MHA anchor: 2 * n_head * head_dim = 2 * 8 * 64 = 1024  (K and V, uncompressed)
# Full-MLA anchor: d_c(128) + n_head * d_rope(8*32=256) = 384  (per-head K^R, this impl)
MHA_KV_CACHE   = 1024
FULL_MLA_KV_CACHE = 384


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run_dir(run_dir: Path) -> tuple:
    """
    Load a single run directory.
    Returns (d_c, seed, metrics_df) or None if incomplete / missing files.
    """
    config_path  = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"

    if not config_path.exists() or not metrics_path.exists():
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    d_c  = config.get("mla_d_c")
    seed = config.get("seed")

    if d_c is None or seed is None:
        warnings.warn(f"Missing mla_d_c or seed in {config_path}, skipping.")
        return None

    df = pd.read_csv(metrics_path)
    # Guard against duplicate iters from crash-resume overlaps
    df = df.drop_duplicates(subset=["iter"], keep="last")
    df["d_c"]  = int(d_c)
    df["seed"] = int(seed)
    return (int(d_c), int(seed), df)


def load_sweep_runs(sweep_dir: Path) -> pd.DataFrame:
    """Load all mla_norope runs from results_sweep/."""
    dfs = []
    if not sweep_dir.exists():
        warnings.warn(f"Sweep directory not found: {sweep_dir}")
        return pd.DataFrame()

    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        result = load_run_dir(run_dir)
        if result is None:
            continue
        d_c, seed, df = result
        df["source"] = "sweep"
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_reused_dc128(results_dir: Path) -> pd.DataFrame:
    """
    Load d_c=128 points from results/mla_norope_s* runs.
    Validates that config.json reports mla_d_c=128 (never parses dir name).
    """
    dfs = []
    for run_dir in sorted(results_dir.glob("mla_norope_s*")):
        if not run_dir.is_dir():
            continue
        result = load_run_dir(run_dir)
        if result is None:
            continue
        d_c, seed, df = result
        if d_c != 128:
            warnings.warn(
                f"Expected mla_d_c=128 in {run_dir}, got {d_c}. Skipping."
            )
            continue
        df["source"] = "reused"
        dfs.append(df)

    if not dfs:
        warnings.warn(f"No valid mla_norope_s* runs found in {results_dir}")
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_reference_runs(results_dir: Path, prefix: str) -> pd.DataFrame:
    """Load MHA or full-MLA reference runs from results/."""
    dfs = []
    for run_dir in sorted(results_dir.glob(f"{prefix}_s*")):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.csv"
        config_path  = run_dir / "config.json"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        df = df.drop_duplicates(subset=["iter"], keep="last")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            df["seed"] = config.get("seed", df.get("seed", 0))
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def final_val_loss(df: pd.DataFrame) -> float:
    """Mean of the last FINAL_N_POINTS eval points by iter."""
    valid = df.dropna(subset=["val_loss"])
    if valid.empty:
        return float("nan")
    last_n = valid.nlargest(FINAL_N_POINTS, "iter")["val_loss"]
    return last_n.mean()


def aggregate_by_dc(combined_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each unique d_c, compute mean/std/n/95%CI of final_val_loss over seeds.
    Returns (aggregated DataFrame sorted by d_c, per-run DataFrame).
    """
    rows = []
    for (d_c, seed), grp in combined_df.groupby(["d_c", "seed"]):
        loss = final_val_loss(grp)
        rows.append({"d_c": d_c, "seed": seed, "final_val_loss": loss})

    if not rows:
        empty = pd.DataFrame(columns=["d_c", "seed", "final_val_loss"])
        return empty, empty

    per_run = pd.DataFrame(rows)

    agg_rows = []
    for d_c, grp in per_run.groupby("d_c"):
        losses = grp["final_val_loss"].dropna()
        n    = len(losses)
        mean = losses.mean()
        std  = losses.std(ddof=1) if n > 1 else 0.0
        if n >= 2:
            ci = t_dist.ppf(0.975, n - 1) * std / np.sqrt(n)
        else:
            ci = 0.0
        agg_rows.append({
            "d_c":  d_c,
            "n":    n,
            "mean": mean,
            "std":  std,
            "ci95": ci,
        })

    agg = pd.DataFrame(agg_rows).sort_values("d_c").reset_index(drop=True)
    return agg, per_run


def reference_mean(df: pd.DataFrame) -> float:
    """Compute mean final val loss across all seeds in a reference DataFrame."""
    if df.empty:
        return float("nan")
    seed_losses = []
    for seed, grp in df.groupby("seed"):
        seed_losses.append(final_val_loss(grp))
    return float(np.nanmean(seed_losses))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dose_response(
    agg: pd.DataFrame,
    per_run: pd.DataFrame,
    mha_mean: float,
    full_mla_mean: float,
    out_dir: Path,
) -> None:
    """
    Dose-response: x=d_c, y=mean final val loss with 95% CI errorbars
    and individual seed scatter. Horizontal dashed line for MHA baseline.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    dc_sorted = agg["d_c"].values
    means     = agg["mean"].values
    cis       = agg["ci95"].values

    ax.errorbar(
        dc_sorted, means, yerr=cis,
        fmt="o-", color="#c44e52", linewidth=2, markersize=7,
        capsize=5, elinewidth=1.5, capthick=1.5,
        label="mla_norope mean +/- 95% CI",
        zorder=3,
    )

    # Individual seed scatter
    if not per_run.empty:
        for _, row in per_run.iterrows():
            ax.scatter(row["d_c"], row["final_val_loss"],
                       color="#c44e52", alpha=0.45, s=30, zorder=4)

    # MHA baseline
    if not np.isnan(mha_mean):
        ax.axhline(mha_mean, color="#4c72b0", linestyle="--", linewidth=1.5,
                   label=f"MHA baseline ({mha_mean:.4f})", zorder=2)

    # Full-MLA reference
    if not np.isnan(full_mla_mean):
        ax.axhline(full_mla_mean, color="#dd8452", linestyle=":", linewidth=1.5,
                   label=f"Full MLA ({full_mla_mean:.4f})", zorder=2)

    ax.set_xlabel("d_c (compression dimension)", fontsize=12)
    ax.set_ylabel("Final Validation Loss\n(mean of last 5 eval points)", fontsize=11)
    ax.set_title("Dose-Response: mla_norope d_c Sweep", fontsize=13)
    ax.set_xticks(dc_sorted)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "sweep_dose_response.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_pareto(
    agg: pd.DataFrame,
    mha_mean: float,
    full_mla_mean: float,
    out_dir: Path,
) -> None:
    """
    Quality-vs-KV-cache Pareto chart.
    x = KV-cache scalars/token/layer (for mla_norope = d_c).
    y = final val loss. Lower-left is better.
    Includes MHA anchor at 1024 and full-MLA anchor at 384.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    # Sweep points: KV cache = d_c for mla_norope
    if not agg.empty:
        ax.scatter(
            agg["d_c"], agg["mean"],
            color="#c44e52", s=80, zorder=4,
            label="mla_norope sweep",
        )
        for _, row in agg.iterrows():
            ax.annotate(
                f"d_c={int(row['d_c'])}",
                xy=(row["d_c"], row["mean"]),
                xytext=(6, 4), textcoords="offset points",
                fontsize=8,
            )
        # CI bars
        ax.errorbar(
            agg["d_c"], agg["mean"], yerr=agg["ci95"],
            fmt="none", color="#c44e52", capsize=4, elinewidth=1.2, capthick=1.2,
            zorder=3,
        )

    # MHA anchor
    if not np.isnan(mha_mean):
        ax.scatter(
            [MHA_KV_CACHE], [mha_mean],
            marker="D", color="#4c72b0", s=90, zorder=4, label="MHA baseline",
        )
        ax.annotate(
            f"MHA\n(d_c={MHA_KV_CACHE})",
            xy=(MHA_KV_CACHE, mha_mean),
            xytext=(-55, 4), textcoords="offset points",
            fontsize=8, color="#4c72b0",
        )

    # Full-MLA anchor
    if not np.isnan(full_mla_mean):
        ax.scatter(
            [FULL_MLA_KV_CACHE], [full_mla_mean],
            marker="s", color="#dd8452", s=90, zorder=4, label="Full MLA",
        )
        ax.annotate(
            f"Full MLA\n(eff. d_c~{FULL_MLA_KV_CACHE})",
            xy=(FULL_MLA_KV_CACHE, full_mla_mean),
            xytext=(6, 4), textcoords="offset points",
            fontsize=8, color="#dd8452",
        )

    ax.set_xlabel("Theoretical KV-cache scalars/token/layer", fontsize=12)
    ax.set_ylabel("Final Validation Loss", fontsize=11)
    ax.set_title("Quality vs KV-cache Pareto  (lower-left = better)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "sweep_pareto.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary_table(agg: pd.DataFrame, mha_mean: float) -> None:
    """Print ASCII-only summary table (no unicode glyphs)."""
    print()
    print("--- d_c Sweep Summary (mla_norope) ---")
    header = f"{'d_c':>6} | {'n':>4} | {'mean':>8} | {'std':>8} | {'95%CI':>8} | {'delta-vs-MHA':>13}"
    print(header)
    print("-" * len(header))
    for _, row in agg.iterrows():
        delta = row["mean"] - mha_mean if not np.isnan(mha_mean) else float("nan")
        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "  n/a  "
        ci_str    = f"+/-{row['ci95']:.4f}" if row["n"] >= 2 else "  n/a  "
        std_str   = f"{row['std']:.4f}" if row["n"] >= 2 else "  n/a  "
        print(
            f"{int(row['d_c']):>6} | {int(row['n']):>4} | {row['mean']:>8.4f} | "
            f"{std_str:>8} | {ci_str:>8} | {delta_str:>13}"
        )
    print()
    if not np.isnan(mha_mean):
        print(f"MHA baseline mean: {mha_mean:.4f}")
    print()


def save_summary_csv(agg: pd.DataFrame, mha_mean: float, out_dir: Path) -> None:
    out = agg.copy()
    out["delta_vs_mha"] = out["mean"] - mha_mean
    path = out_dir / "sweep_summary.csv"
    out.to_csv(path, index=False)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze d_c compression sweep for mla_norope."
    )
    parser.add_argument("--results_dir", default="results",
                        help="Dir with existing 2x2 ablation results (for d_c=128 reuse and MHA/MLA refs)")
    parser.add_argument("--sweep_dir",   default="results_sweep",
                        help="Dir with new sweep runs (d_c != 128)")
    parser.add_argument("--out_dir",     default="plots",
                        help="Output directory for plots and CSV")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    sweep_dir   = Path(args.sweep_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data -----------------------------------------------------------
    print(f"Loading new sweep runs from: {sweep_dir}")
    sweep_df = load_sweep_runs(sweep_dir)
    if not sweep_df.empty:
        print(f"  Loaded {len(sweep_df):,} rows "
              f"(d_c values: {sorted(sweep_df['d_c'].unique())})")
    else:
        print("  No sweep data found (runs not yet completed).")

    print(f"Loading reused d_c=128 from: {results_dir}/mla_norope_s*")
    dc128_df = load_reused_dc128(results_dir)
    if not dc128_df.empty:
        print(f"  Loaded {len(dc128_df):,} rows "
              f"(seeds: {sorted(dc128_df['seed'].unique())})")
    else:
        print("  No reused d_c=128 data found.")

    print(f"Loading MHA reference from: {results_dir}/mha_s*")
    mha_df = load_reference_runs(results_dir, "mha")
    print(f"Loading full-MLA reference from: {results_dir}/mla_s*")
    full_mla_df = load_reference_runs(results_dir, "mla")

    # --- Combine sweep + reused d_c=128 --------------------------------------
    all_parts = [p for p in [sweep_df, dc128_df] if not p.empty]
    if not all_parts:
        print("No mla_norope data available at all. Run the sweep first.")
        return

    combined = pd.concat(all_parts, ignore_index=True)

    # --- Reference means -----------------------------------------------------
    mha_mean      = reference_mean(mha_df)
    full_mla_mean = reference_mean(full_mla_df)

    if not np.isnan(mha_mean):
        print(f"MHA baseline mean final val loss: {mha_mean:.4f}")
    else:
        print("MHA reference not found; delta column will be NaN.")

    if not np.isnan(full_mla_mean):
        print(f"Full-MLA reference mean final val loss: {full_mla_mean:.4f}")

    # --- Aggregate -----------------------------------------------------------
    agg, per_run = aggregate_by_dc(combined)

    if agg.empty:
        print("Aggregation produced no rows; check that metrics.csv files have val_loss.")
        return

    # --- Summary table -------------------------------------------------------
    print_summary_table(agg, mha_mean)

    # --- Plots ---------------------------------------------------------------
    plot_dose_response(agg, per_run, mha_mean, full_mla_mean, out_dir)
    plot_pareto(agg, mha_mean, full_mla_mean, out_dir)

    # --- CSV -----------------------------------------------------------------
    save_summary_csv(agg, mha_mean, out_dir)


if __name__ == "__main__":
    main()
