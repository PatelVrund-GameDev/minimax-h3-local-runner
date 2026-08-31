@echo off
setlocal enabledelayedexpansion
title MiniMax H3 and Wan2GP Local Runner

cd /d "%~dp0"

echo ==============================================================================
echo              MINIMAX H3 AND WAN2GP - LOCAL STUDIO RUNNER
echo ==============================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [!] Error: Virtual environment not found in .venv.
    echo [*] Please run 'setup_windows.bat' first to install the required components.
    echo.
    pause
    exit /b 1
)

:: Set Cache and Temporary Paths to local directory on D: drive (prevents C: disk full)
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

:: GPU Memory Optimization
set "CUDA_MODULE_LOADING=LAZY"

echo [*] Cache Directory:  %HF_HOME%
echo [*] Temporary Path:   %TMPDIR%
echo.
echo [*] Initializing Wan2GP engine and GPU acceleration kernels...
echo [*] The server is starting. Your browser will automatically open to:
echo     http://127.0.0.1:7860
echo ==============================================================================
echo.

:: Launch background watcher to automatically open browser once the web server is ready
start /b powershell -NoProfile -ExecutionPolicy Bypass -Command "$port = 7860; $maxWait = 120; $elapsed = 0; while ($elapsed -lt $maxWait) { try { $client = New-Object System.Net.Sockets.TcpClient('127.0.0.1', $port); if ($client.Connected) { $client.Close(); Start-Process 'http://127.0.0.1:7860'; break } } catch { Start-Sleep -Seconds 1; $elapsed++ } }"

cd /d "%BASE_DIR%wan2gp_core"
"%BASE_DIR%.venv\Scripts\python.exe" wgp.py %*

if %errorlevel% neq 0 (
    echo.
    echo [!] Wan2GP encountered an error. Check the logs above.
    pause
)
