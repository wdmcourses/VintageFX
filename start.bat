@echo off
chcp 65001 >nul 2>&1
title VintageFX
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo ==================================================
echo   VintageFX - era effects for your audio
echo   Starting local server, browser will open...
echo ==================================================

"python\python.exe" "app\server.py"
if errorlevel 1 (
  echo.
  echo [ERROR] Failed to start. Press any key to see the error.
  pause >nul
)
