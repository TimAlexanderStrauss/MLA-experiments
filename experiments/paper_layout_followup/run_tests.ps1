$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
$Python = Join-Path $ScriptDir "..\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Python environment not found at $Python. Follow README.md."
}

& $Python "test_correctness.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python "test_reference.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "All follow-up tests passed."
