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
    throw "Python environment not found at $Python. Follow README.md."
}
if (-not (Test-Path (Join-Path $DataDir "train.bin")) -or
    -not (Test-Path (Join-Path $DataDir "val.bin"))) {
    throw "FineWeb-Edu train.bin/val.bin not found at $DataDir. Follow README.md."
}

$Modes = @("mha", "mha_rope", "mla_norope", "mla")
$Seeds = @("42", "123", "456")
if ($Mode -ne "" -and $Modes -notcontains $Mode) {
    throw "Unknown mode '$Mode'. Choose: $($Modes -join ', ')"
}
if ($Seed -ne "" -and $Seeds -notcontains $Seed) {
    throw "Unknown seed '$Seed'. Choose: $($Seeds -join ', ')"
}

# Balanced order: every mode appears early, middle and late across seeds.
$RunPlan = @(
    [PSCustomObject]@{ Mode = "mha";        Seed = "42"  },
    [PSCustomObject]@{ Mode = "mha_rope";   Seed = "42"  },
    [PSCustomObject]@{ Mode = "mla_norope"; Seed = "42"  },
    [PSCustomObject]@{ Mode = "mla";        Seed = "42"  },
    [PSCustomObject]@{ Mode = "mha_rope";   Seed = "123" },
    [PSCustomObject]@{ Mode = "mla_norope"; Seed = "123" },
    [PSCustomObject]@{ Mode = "mla";        Seed = "123" },
    [PSCustomObject]@{ Mode = "mha";        Seed = "123" },
    [PSCustomObject]@{ Mode = "mla_norope"; Seed = "456" },
    [PSCustomObject]@{ Mode = "mla";        Seed = "456" },
    [PSCustomObject]@{ Mode = "mha";        Seed = "456" },
    [PSCustomObject]@{ Mode = "mha_rope";   Seed = "456" }
)

function Write-Log($Message) {
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$Timestamp] $Message"
}

$Ran = 0
$Skipped = 0
foreach ($Run in $RunPlan) {
    if ($Mode -ne "" -and $Run.Mode -ne $Mode) { continue }
    if ($Seed -ne "" -and $Run.Seed -ne $Seed) { continue }
    $OutDir = "results/$($Run.Mode)_s$($Run.Seed)"
    $DoneFlag = "$OutDir/DONE"
    if (Test-Path $DoneFlag) {
        Write-Log "SKIP $($Run.Mode) seed=$($Run.Seed) (DONE exists)"
        $Skipped++
        continue
    }

    Write-Log "START $($Run.Mode) seed=$($Run.Seed)"
    & $Python "train.py" `
        "--attn_mode=$($Run.Mode)" `
        "--seed=$($Run.Seed)" `
        "--out_dir=$OutDir" `
        "--data_dir=$DataDir" `
        "--max_iters=15300" `
        "--batch_size=16" `
        "--grad_accum=4" `
        "--lr=6e-4" `
        "--min_lr=6e-5" `
        "--warmup_iters=2000" `
        "--eval_interval=500" `
        "--eval_iters=600" `
        "--save_interval=1000" `
        "--dtype=bfloat16" `
        "--no-compile"
    if ($LASTEXITCODE -ne 0) {
        Write-Log "ERROR $($Run.Mode) seed=$($Run.Seed), exit=$LASTEXITCODE"
        exit $LASTEXITCODE
    }
    New-Item -ItemType File -Path $DoneFlag -Force | Out-Null
    Write-Log "DONE $($Run.Mode) seed=$($Run.Seed)"
    $Ran++
}

Write-Log "Finished. ran=$Ran skipped=$Skipped"
