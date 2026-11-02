$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonExe = Join-Path $PSScriptRoot 'venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Chua co venv. Chay setup_local.ps1 truoc.' }
$env:PYTHONIOENCODING = 'utf-8'
$env:YOLO_CONFIG_DIR = Join-Path $PSScriptRoot 'local_runs/config'
$env:YOLO_AUTOINSTALL = 'false'
Write-Host 'EcoVision: http://127.0.0.1:8000 - Nhan Ctrl+C de dung.'
& $pythonExe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
exit $LASTEXITCODE
