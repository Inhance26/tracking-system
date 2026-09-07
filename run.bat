@echo off
REM Start the tracker with YOLO on the bundled CCTV footage and open the
REM dashboard. Run setup.bat first if you haven't.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo   No virtual environment found. Run setup.bat first.
  echo.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" app.py --open %*

pause
