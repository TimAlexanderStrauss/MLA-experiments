$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
$Python = Join-Path $ScriptDir "..\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Python environment not found at $Python. See README.md."
}
& $Python "smoke_test_gpu.py"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
& $Python "test_gpu_capacity.py"
exit $LASTEXITCODE
