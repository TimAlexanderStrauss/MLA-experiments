# =============================================================================
# run_experiments.ps1 - Launch all 12 runs of the 2x2 MLA ablation study
#
# 4 conditions x 3 seeds = 12 sequential runs
#
# Usage:
#   cd experiments
#   .\run_experiments.ps1                     # run all 12
#   .\run_experiments.ps1 -Mode mha           # run only mha condition (3 seeds)
#   .\run_experiments.ps1 -Mode mla -Seed 42  # run only mla with seed 42
#
# Checkpoints are written to results/<mode>_s<seed>/checkpoint.pt
# Interrupted runs resume automatically from the last checkpoint.
# =============================================================================

param(
    [string]$Mode = "",
    [string]$Seed = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$Python = Join-Path $ScriptDir ".venv\Scripts\python.exe"

# ---- Configuration (must match design doc section 4) -------------------------
$Modes = @("mha", "mha_rope", "mla_norope", "mla")
$Seeds = @("42", "123", "456")

$MAX_ITERS     = 15300
$BATCH_SIZE    = 16
$GRAD_ACCUM    = 4
$LR            = "6e-4"
$MIN_LR        = "6e-5"
$WARMUP        = 2000
$EVAL_INTERVAL = 500
$EVAL_ITERS    = 600
$SAVE_INTERVAL = 5000

# ------------------------------------------------------------------------------
function Write-Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] $msg"
}

$runCount  = 0
$skipCount = 0

foreach ($m in $Modes) {
    if ($Mode -ne "" -and $m -ne $Mode) { continue }

    foreach ($s in $Seeds) {
        if ($Seed -ne "" -and $s -ne $Seed) { continue }

        $outDir   = "results/${m}_s${s}"
        $doneFlag = "${outDir}/DONE"

        if (Test-Path $doneFlag) {
            Write-Log "SKIP  $m seed=$s  (already completed: $doneFlag)"
            $skipCount++
            continue
        }

        Write-Log "START $m seed=$s  -> $outDir"
        $startTime = Get-Date

        $trainArgs = @(
            "train.py",
            "--attn_mode=$m",
            "--seed=$s",
            "--out_dir=$outDir",
            "--max_iters=$MAX_ITERS",
            "--batch_size=$BATCH_SIZE",
            "--grad_accum=$GRAD_ACCUM",
            "--lr=$LR",
            "--min_lr=$MIN_LR",
            "--warmup_iters=$WARMUP",
            "--eval_interval=$EVAL_INTERVAL",
            "--eval_iters=$EVAL_ITERS",
            "--save_interval=$SAVE_INTERVAL",
            "--dtype=bfloat16",
            "--no-compile"
        )

        & $Python @trainArgs

        if ($LASTEXITCODE -ne 0) {
            Write-Log "ERROR $m seed=$s exited with code $LASTEXITCODE - aborting."
            exit 1
        }

        $elapsed = [int]((Get-Date) - $startTime).TotalSeconds
        Write-Log "DONE  $m seed=$s  elapsed=${elapsed}s"

        New-Item -ItemType File -Path $doneFlag -Force | Out-Null
        $runCount++
    }
}

Write-Log "Finished.  ran=$runCount  skipped=$skipCount"
