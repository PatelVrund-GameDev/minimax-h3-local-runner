@echo off
setlocal enabledelayedexpansion
title MiniMax H3 Local Runner - 1-Click Updater

cd /d "%~dp0"

echo ==============================================================================
echo           MINIMAX H3 LOCAL RUNNER AND WAN2GP - 1-CLICK UPDATER
echo ==============================================================================
echo.

echo [*] Pulling latest updates for MiniMax H3 Local Runner...
git pull origin main

if exist "wan2gp_core" (
    echo.
    echo [*] Pulling latest upstream Wan2GP updates...
    pushd wan2gp_core
    git pull origin main
    popd
)

if exist ".venv\Scripts\python.exe" (
    echo.
    echo [*] Updating Python dependencies...
    if exist "wan2gp_core\requirements.txt" (
        ".venv\Scripts\python.exe" -m pip install -r wan2gp_core\requirements.txt -U
    )
)

echo.
echo ==============================================================================
echo [OK] UPDATE COMPLETE!
echo ==============================================================================
echo.
pause
