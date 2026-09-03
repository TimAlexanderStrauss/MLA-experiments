param(
    [string]$Backbone = "",
    [string]$Attention = "",
    [string]$Seed = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
$Python = Join-Path $ScriptDir "..\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Python environment not found at $Python. See README.md."
}
if (-not (Test-Path "gpu_profile.json")) {
    & $Python "benchmark_gpu.py"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$Arguments = @("run_experiments.py")
if ($Backbone -ne "") { $Arguments += "--backbone=$Backbone" }
if ($Attention -ne "") { $Arguments += "--attn_mode=$Attention" }
if ($Seed -ne "") { $Arguments += "--seed=$Seed" }
& $Python @Arguments
exit $LASTEXITCODE
