@echo off
cd /d "%~dp0"
echo Unified launcher: CCTV tracker + dashboard
echo Press Ctrl+C to stop both.
echo.
python run_system.py
pause
