# =============================================================================
# run_sweep.ps1 - d_c compression-strength sweep for mla_norope
#
# Sweeps mla_d_c over {64, 256, 384, 512} x seeds {42, 123} = 8 new runs.
# d_c=128 is reused from the existing results/mla_norope_s* runs.
#
# Usage:
#   cd experiments
#   .\run_sweep.ps1                      # run all 8 new runs
#   .\run_sweep.ps1 -Dc 256             # run only d_c=256 (all seeds)
#   .\run_sweep.ps1 -Dc 256 -Seed 42    # run only d_c=256 seed=42
#
# Outputs go to results_sweep/mla_norope_dc<dc>_s<seed>/
# (results/ is left untouched so the 2x2 analysis remains valid)
# =============================================================================

param(
    [string]$Dc   = "",
    [string]$Seed = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$Python = Join-Path $ScriptDir ".venv\Scripts\python.exe"

# ---- Sweep configuration ----------------------------------------------------
# d_c=128 is deliberately excluded here; it is reused from results/mla_norope_s*
$DcValues = @("64", "256", "384", "512")
$Seeds    = @("42", "123")

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

foreach ($dc in $DcValues) {
    if ($Dc -ne "" -and $dc -ne $Dc) { continue }

    foreach ($s in $Seeds) {
        if ($Seed -ne "" -and $s -ne $Seed) { continue }

        $outDir   = "results_sweep/mla_norope_dc${dc}_s${s}"
        $doneFlag = "${outDir}/DONE"

        if (Test-Path $doneFlag) {
            Write-Log "SKIP  mla_norope d_c=$dc seed=$s  (already completed: $doneFlag)"
            $skipCount++
            continue
        }

        Write-Log "START mla_norope d_c=$dc seed=$s  -> $outDir"
        $startTime = Get-Date

        $trainArgs = @(
            "train.py",
            "--attn_mode=mla_norope",
            "--seed=$s",
            "--out_dir=$outDir",
            "--mla_d_c=$dc",
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
            Write-Log "ERROR mla_norope d_c=$dc seed=$s exited with code $LASTEXITCODE - aborting."
            exit 1
        }

        $elapsed = [int]((Get-Date) - $startTime).TotalSeconds
        Write-Log "DONE  mla_norope d_c=$dc seed=$s  elapsed=${elapsed}s"

        New-Item -ItemType File -Path $doneFlag -Force | Out-Null
        $runCount++
    }
}

Write-Log "Finished.  ran=$runCount  skipped=$skipCount"
