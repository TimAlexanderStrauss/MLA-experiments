#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh — Launch all 12 runs of the 2×2 MLA ablation study
#
# 4 conditions × 3 seeds = 12 sequential runs
# Estimated ~5.5 h per run on RTX 5070 → ~66 h total (4-5 days wall-clock)
#
# Usage:
#   cd experiments
#   bash run_experiments.sh               # run all 12
#   bash run_experiments.sh mha           # run only mha condition (3 seeds)
#   bash run_experiments.sh mla 42        # run only mla with seed 42
#
# Checkpoints are written to results/<mode>_s<seed>/checkpoint.pt
# Interrupted runs resume automatically from the last checkpoint.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Configuration (edit here if needed) ------------------------------------
MODES=(mha mha_rope mla_norope mla)
SEEDS=(42 123 456)

# Training hyperparameters (must match design doc §4)
MAX_ITERS=15300      # ~500M tokens at batch=64, seq=512
BATCH_SIZE=16        # micro-batch size
GRAD_ACCUM=4         # effective batch = 16 × 4 = 64
LR=6e-4
MIN_LR=6e-5
WARMUP=2000
EVAL_INTERVAL=500
EVAL_ITERS=600          # ~5M tokens per eval (covers the full hold-out)
SAVE_INTERVAL=5000

# ---------------------------------------------------------------------------
# Optional filter from command-line arguments
FILTER_MODE="${1:-}"    # e.g. "mha" — if set, only run this mode
FILTER_SEED="${2:-}"    # e.g. "42"  — if set, only run this seed

# ---------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_count=0
skip_count=0

for MODE in "${MODES[@]}"; do
    # Skip if mode filter is active and doesn't match
    if [[ -n "$FILTER_MODE" && "$MODE" != "$FILTER_MODE" ]]; then
        continue
    fi

    for SEED in "${SEEDS[@]}"; do
        # Skip if seed filter is active and doesn't match
        if [[ -n "$FILTER_SEED" && "$SEED" != "$FILTER_SEED" ]]; then
            continue
        fi

        OUT_DIR="results/${MODE}_s${SEED}"
        DONE_FLAG="${OUT_DIR}/DONE"

        if [[ -f "$DONE_FLAG" ]]; then
            log "SKIP  ${MODE} seed=${SEED}  (already completed: ${DONE_FLAG})"
            ((skip_count++)) || true
            continue
        fi

        log "START ${MODE} seed=${SEED}  → ${OUT_DIR}"
        START_TS=$(date +%s)

        python train.py \
            --attn_mode="$MODE" \
            --seed="$SEED" \
            --out_dir="$OUT_DIR" \
            --max_iters="$MAX_ITERS" \
            --batch_size="$BATCH_SIZE" \
            --grad_accum="$GRAD_ACCUM" \
            --lr="$LR" \
            --min_lr="$MIN_LR" \
            --warmup_iters="$WARMUP" \
            --eval_interval="$EVAL_INTERVAL" \
            --eval_iters="$EVAL_ITERS" \
            --save_interval="$SAVE_INTERVAL" \
            --dtype=bfloat16 \
            --no-compile   # all reported runs used --no-compile (see configs);
                           # keep this in sync with run_experiments.ps1

        END_TS=$(date +%s)
        ELAPSED=$(( END_TS - START_TS ))
        log "DONE  ${MODE} seed=${SEED}  elapsed=${ELAPSED}s"

        # Mark as completed so re-runs skip this
        touch "$DONE_FLAG"
        ((run_count++)) || true
    done
done

log "Finished.  ran=${run_count}  skipped=${skip_count}"
