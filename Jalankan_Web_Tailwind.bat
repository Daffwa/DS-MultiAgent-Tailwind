@echo off
title Data Science Multi-Agent Solver (Tailwind CSS + FastAPI)
echo =========================================================================
echo   Memulai Data Science Multi-Agent Solver (Tailwind CSS + FastAPI)...
echo   Browser akan otomatis terbuka ke tampilan modern Next.js/Tailwind!
echo =========================================================================
echo.

cd /d "%~dp0"
python run_server.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Peringatan] Terjadi kendala saat menjalankan python.
    pause
)
