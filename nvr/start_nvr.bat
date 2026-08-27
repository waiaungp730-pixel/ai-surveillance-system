@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

echo === Real-time NVR (go2rtc WebRTC) ===
py -3.12 "%~dp0generate_configs.py"
if errorlevel 1 exit /b 1
if not exist "%~dp0logs" mkdir "%~dp0logs"

REM resolve_ffmpeg already ran inside generate_configs.py — do not hard-fail on PATH
where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo NOTE: ffmpeg not on PATH — using auto-detected path from generate_configs.py
)

set "GO2RTC_BIN=go2rtc"
where go2rtc >nul 2>&1
if errorlevel 1 (
  if exist "%~dp0go2rtc.exe" (
    set "GO2RTC_BIN=%~dp0go2rtc.exe"
  ) else (
    echo ERROR: go2rtc not found.
    echo Download: https://github.com/AlexxIT/go2rtc/releases
    echo Put go2rtc.exe in nvr\ or on PATH.
    exit /b 1
  )
)

echo Starting go2rtc on :1984 ...
start "go2rtc" /min "%GO2RTC_BIN%" -config "%~dp0go2rtc.yaml"
REM Avoid `timeout` (fails under some redirected shells)
ping -n 3 127.0.0.1 >nul

where nginx >nul 2>&1
if errorlevel 1 (
  echo nginx optional — use http://127.0.0.1:5000 + go2rtc :1984
) else (
  nginx -c "%~dp0nginx.conf" -p "%~dp0"
  echo Open http://127.0.0.1:8088  ^(Live Cameras = real-time WebRTC^)
)

echo go2rtc UI: http://127.0.0.1:1984/
endlocal
