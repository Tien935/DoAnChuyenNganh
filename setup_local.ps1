$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Assert-Command([string]$step) { if ($LASTEXITCODE -ne 0) { throw "That bai: $step" } }
$pythonExe = Join-Path $PSScriptRoot 'venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { python -m venv venv; Assert-Command 'Tao venv' }
& $pythonExe -c 'import torch, torchvision' 2>$null
if ($LASTEXITCODE -ne 0) {
    & $pythonExe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    Assert-Command 'Cai PyTorch'
}
& $pythonExe -m pip install -r requirements-local.txt
Assert-Command 'Cai thu vien'
if (-not (Test-Path -LiteralPath 'yolov12/pyproject.toml')) {
    git clone https://github.com/sunsmarterjie/yolov12.git yolov12
    Assert-Command 'Tai ma nguon YOLOv12'
    git -C yolov12 checkout 64d5ffc6db35a1bde446689ab8450ecd43475e00
    Assert-Command 'Chon phien ban YOLOv12'
}
& $pythonExe -m pip install --no-deps --no-build-isolation -e ./yolov12
Assert-Command 'Cai YOLOv12'
if (-not (Test-Path -LiteralPath 'dataset/test/images')) {
    git fetch origin
    Assert-Command 'Cap nhat Git'
    git restore --source=origin/Tu --worktree -- dataset
    Assert-Command 'Lay dataset tu nhanh Tu'
}
New-Item -ItemType Directory -Force -Path models | Out-Null
if (-not (Test-Path -LiteralPath 'models/yolov12n-seg.pt')) {
    Invoke-WebRequest -Uri 'https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12n-seg.pt' -OutFile 'models/yolov12n-seg.pt'
}
Write-Host 'Cai dat xong. Chay .\start_local.ps1. Chep trong so rac cua nhom vao models/best.pt.'
