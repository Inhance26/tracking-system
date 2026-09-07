@echo off
REM One-time setup: virtual environment + all dependencies + a YOLO self-test.
REM Double-click this, or run it from the VS Code terminal.

cd /d "%~dp0"
echo.
echo   Setting up the tracker in %CD%
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo   Python was not found on your PATH.
  echo   Install Python 3.12 from python.org and tick "Add python.exe to PATH".
  echo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo   [1/3] Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 goto fail
) else (
  echo   [1/3] Virtual environment already exists.
)

echo   [2/3] Installing dependencies. Ultralytics pulls in PyTorch,
echo         so this downloads 2-3 GB and can take 10+ minutes.
echo.
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail

echo.
echo   [3/3] Testing YOLO on the bundled footage...
echo.
".venv\Scripts\python.exe" check_setup.py

echo.
echo   Setup finished. Start the tracker with run.bat
echo.
pause
exit /b 0

:fail
echo.
echo   Setup failed - see the error above.
echo.
pause
exit /b 1
