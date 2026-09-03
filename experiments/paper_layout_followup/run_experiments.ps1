# Run the clean 3-condition x 3-seed paper-layout sensitivity study on Windows.

param(
    [string]$Mode = "",
    [string]$Seed = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$Python = Join-Path $ScriptDir "..\.venv\Scripts\python.exe"
$DataDir = Join-Path $ScriptDir "..\data\fineweb_edu"

if (-not (Test-Path $Python)) {
    throw "Python environment not found at $Python. Follow README.md section 'Windows-Vorbereitung'."
}
if (-not (Test-Path (Join-Path $DataDir "train.bin")) -or
    -not (Test-Path (Join-Path $DataDir "val.bin"))) {
    throw "FineWeb-Edu train.bin/val.bin not found at $DataDir. Follow README.md section 'Daten'."
}

$Modes = @("mha", "mla_current", "mla_deepseek")
$Seeds = @("42", "123", "456")

# Balanced order: every condition occurs once in an early, middle and late
# position. DONE flags still make restarts idempotent.
$RunPlan = @(
    [PSCustomObject]@{ Mode = "mha";          Seed = "42"  },
    [PSCustomObject]@{ Mode = "mla_current";  Seed = "42"  },
    [PSCustomObject]@{ Mode = "mla_deepseek"; Seed = "42"  },
    [PSCustomObject]@{ Mode = "mla_current";  Seed = "123" },
    [PSCustomObject]@{ Mode = "mla_deepseek"; Seed = "123" },
    [PSCustomObject]@{ Mode = "mha";          Seed = "123" },
    [PSCustomObject]@{ Mode = "mla_deepseek"; Seed = "456" },
    [PSCustomObject]@{ Mode = "mha";          Seed = "456" },
    [PSCustomObject]@{ Mode = "mla_current";  Seed = "456" }
)

if ($Mode -ne "" -and $Modes -notcontains $Mode) {
    throw "Unknown mode '$Mode'. Choose: $($Modes -join ', ')"
}
if ($Seed -ne "" -and $Seeds -notcontains $Seed) {
    throw "Unknown seed '$Seed'. Choose: $($Seeds -join ', ')"
}

$MaxIters = 15300
$BatchSize = 16
$GradAccum = 4
$LearningRate = "6e-4"
$MinLearningRate = "6e-5"
$Warmup = 2000
$EvalInterval = 500
$EvalIters = 600
$SaveInterval = 5000

function Write-Log($Message) {
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$Timestamp] $Message"
}

$RunCount = 0
$SkipCount = 0

foreach ($Run in $RunPlan) {
    $CurrentMode = $Run.Mode
    $CurrentSeed = $Run.Seed
    if ($Mode -ne "" -and $CurrentMode -ne $Mode) { continue }
    if ($Seed -ne "" -and $CurrentSeed -ne $Seed) { continue }

    $OutDir = "results/${CurrentMode}_s${CurrentSeed}"
    $DoneFlag = "${OutDir}/DONE"

    if (Test-Path $DoneFlag) {
        Write-Log "SKIP $CurrentMode seed=$CurrentSeed (already complete)"
        $SkipCount++
        continue
    }

    Write-Log "START $CurrentMode seed=$CurrentSeed -> $OutDir"
    $StartTime = Get-Date
    $Arguments = @(
        "train.py",
        "--attn_mode=$CurrentMode",
        "--seed=$CurrentSeed",
        "--out_dir=$OutDir",
        "--data_dir=$DataDir",
        "--max_iters=$MaxIters",
        "--batch_size=$BatchSize",
        "--grad_accum=$GradAccum",
        "--lr=$LearningRate",
        "--min_lr=$MinLearningRate",
        "--warmup_iters=$Warmup",
        "--eval_interval=$EvalInterval",
        "--eval_iters=$EvalIters",
        "--save_interval=$SaveInterval",
        "--dtype=bfloat16",
        "--no-compile"
    )

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Log "ERROR $CurrentMode seed=$CurrentSeed exited with code $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    New-Item -ItemType File -Path $DoneFlag -Force | Out-Null
    $Elapsed = [int]((Get-Date) - $StartTime).TotalSeconds
    Write-Log "DONE $CurrentMode seed=$CurrentSeed elapsed=${Elapsed}s"
    $RunCount++
}

Write-Log "Finished. ran=$RunCount skipped=$SkipCount"
