@echo off
cd /d "%~dp0"
taskkill /FI "WINDOWTITLE eq go2rtc*" /F >nul 2>&1
where nginx >nul 2>&1 && nginx -s quit -p "%~dp0" >nul 2>&1
echo Stopped go2rtc / nginx (best effort).
