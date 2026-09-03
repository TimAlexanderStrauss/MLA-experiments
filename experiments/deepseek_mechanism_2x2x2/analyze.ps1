$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
$Python = Join-Path $ScriptDir "..\.venv\Scripts\python.exe"
& $Python "analyze_results.py" "--results_dir=results" "--out_dir=plots"
exit $LASTEXITCODE
