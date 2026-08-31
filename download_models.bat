@echo off
setlocal enabledelayedexpansion
title MiniMax H3 Model Downloader

cd /d "%~dp0"

echo ==============================================================================
echo                      MINIMAX H3 MODEL DOWNLOADER WIZARD
echo ==============================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Using Python: !PYTHON_EXE!
echo.

!PYTHON_EXE! download_models.py

echo.
pause
