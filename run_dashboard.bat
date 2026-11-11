@echo off
REM One-click launcher for the Bharat-TTC judge dashboard (Windows only;
REM on macOS/Linux follow the manual steps in README.md).
REM Uses .venv inside this folder and creates it on first run.
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo First run: creating .venv and installing requirements, this takes 1-2 minutes...
  py -3 -m venv .venv 2>nul || python -m venv .venv || (echo Python 3.10+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
  "%PY%" -m pip install --upgrade pip
  "%PY%" -m pip install -r requirements.txt || (echo Installing requirements failed. & pause & exit /b 1)
)
echo Starting Bharat-TTC dashboard on http://127.0.0.1:8000  (close this window to stop)
if not defined BHARAT_NO_BROWSER start "" cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8000"
"%PY%" -m uvicorn web.server:app --host 127.0.0.1 --port 8000 --log-level warning
