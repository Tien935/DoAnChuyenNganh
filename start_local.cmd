@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set YOLO_AUTOINSTALL=false
set YOLO_CONFIG_DIR=%~dp0local_runs\config
echo EcoVision: http://127.0.0.1:8000
echo Keep this window open. Press Ctrl+C to stop.
"%~dp0venv\Scripts\python.exe" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
pause
