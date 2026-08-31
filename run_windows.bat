@echo off
setlocal enabledelayedexpansion
title MiniMax H3 Local Video Agent Launcher

cd /d "%~dp0"

echo ==============================================================================
echo              MINIMAX H3 LOCAL VIDEO AGENT - RUNNER
echo ==============================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [!] Error: Virtual environment not found in .venv.
    echo [*] Please run 'setup_windows.bat' first to install the required components.
    echo.
    pause
    exit /b 1
)

:: Set Cache & Temporary Paths to local directory on D: drive (prevents C: disk full)
set "BASE_DIR=%~dp0"
set "HF_HOME=%BASE_DIR%hf_cache"
set "TORCH_HOME=%BASE_DIR%torch_cache"
set "TMPDIR=%BASE_DIR%tmp"
set "TEMP=%BASE_DIR%tmp"
set "TMP=%BASE_DIR%tmp"

if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%TORCH_HOME%" mkdir "%TORCH_HOME%"
if not exist "%TMPDIR%" mkdir "%TMPDIR%"
if not exist "%BASE_DIR%wan2gp_core\ckpts" mkdir "%BASE_DIR%wan2gp_core\ckpts"
if not exist "%BASE_DIR%outputs" mkdir "%BASE_DIR%outputs"

:: GPU Memory & CUDA Allocator Optimization
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "CUDA_MODULE_LOADING=LAZY"

echo [*] Cache Directory:  %HF_HOME%
echo [*] Temporary Path:   %TMPDIR%
echo [*] Python Executable: %BASE_DIR%.venv\Scripts\python.exe
echo.
echo [*] Starting Wan2GP with MiniMax H3 support...
echo [*] Web UI will open in your browser at http://127.0.0.1:7860
echo.

cd /d "%BASE_DIR%wan2gp_core"
"%BASE_DIR%.venv\Scripts\python.exe" wgp.py %*

if %errorlevel% neq 0 (
    echo.
    echo [!] Wan2GP encountered an error. Check the logs above.
    pause
)
