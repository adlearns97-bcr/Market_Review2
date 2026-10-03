@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Virtual environment missing. Run:  python -m venv .venv ^&^& .venv\Scripts\python -m pip install -r requirements.txt
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" main.py %*
