$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
$Python = Join-Path $ScriptDir "..\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Python environment not found at $Python. Follow README.md."
}

& $Python "analyze_results.py" "--results_dir=results" "--out_dir=plots" "--dense_results_dir=../results"
exit $LASTEXITCODE
