$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
$Python = Join-Path $ScriptDir "..\.venv\Scripts\python.exe"
$DataDir = Join-Path $ScriptDir "..\data\fineweb_edu"

if (-not (Test-Path $Python)) { throw "Python environment not found at $Python" }
if (-not (Test-Path (Join-Path $DataDir "train.bin"))) {
    throw "Dataset not found at $DataDir"
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
foreach ($Mode in @("mha", "mla_current", "mla_deepseek")) {
    Write-Host "Smoke test: $Mode"
    & $Python "train.py" `
        "--attn_mode=$Mode" `
        "--seed=42" `
        "--out_dir=smoke_results/${Timestamp}_${Mode}" `
        "--data_dir=$DataDir" `
        "--block_size=128" `
        "--batch_size=2" `
        "--grad_accum=1" `
        "--max_iters=20" `
        "--warmup_iters=5" `
        "--eval_interval=10" `
        "--eval_iters=2" `
        "--save_interval=20" `
        "--dtype=bfloat16" `
        "--no-compile"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "All three GPU smoke tests completed."
