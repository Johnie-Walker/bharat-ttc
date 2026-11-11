@echo off
REM One-click launcher for the Bharat-TTC judge dashboard.
REM Uses .venv in this folder, else ..\.venv; creates .venv on first run if neither exists.
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" if exist "..\.venv\Scripts\python.exe" set "PY=..\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo First run: creating .venv and installing requirements...
  py -3 -m venv .venv || python -m venv .venv || (echo Python 3.10+ is required. & pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo Install failed. & pause & exit /b 1)
  set "PY=.venv\Scripts\python.exe"
)
echo Starting Bharat-TTC dashboard on http://127.0.0.1:8000  (close this window to stop)
start "" cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8000"
"%PY%" -m uvicorn web.server:app --host 127.0.0.1 --port 8000 --log-level warning
