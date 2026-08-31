@echo off
setlocal enabledelayedexpansion
title MiniMax H3 / Wan2GP 1-Click Installer (Windows)

cd /d "%~dp0"

echo ==============================================================================
echo            MINIMAX H3 & WAN2GP - 1-CLICK WINDOWS SETUP
echo ==============================================================================
echo.

:: 1. Check NVIDIA GPU
where nvidia-smi >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] WARNING: nvidia-smi not found. Ensure you have an NVIDIA GPU and drivers installed.
) else (
    echo [✓] NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
)
echo.

:: 2. Setup Python Environment (.venv) using uv or python
set "USE_UV=0"
where uv >nul 2>nul
if %errorlevel% equ 0 (
    set "USE_UV=1"
    echo [✓] Found 'uv' package manager. Using fast uv environment creation.
    if not exist ".venv" (
        echo [*] Creating Python 3.11 virtual environment with uv...
        uv venv --python 3.11 .venv
    )
) else (
    echo [*] 'uv' not found. Falling back to standard Python.
    if not exist ".venv" (
        echo [*] Creating Python virtual environment (.venv)...
        python -m venv .venv
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo [!] Error: Failed to create virtual environment in .venv.
    pause
    exit /b 1
)

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "VENV_PIP=%~dp0.venv\Scripts\pip.exe"

echo [✓] Virtual environment ready: !VENV_PY!
echo.

:: 3. Set temporary cache paths to local drive
set "HF_HOME=%~dp0hf_cache"
set "TORCH_HOME=%~dp0torch_cache"
set "TMPDIR=%~dp0tmp"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%TORCH_HOME%" mkdir "%TORCH_HOME%"
if not exist "%TMPDIR%" mkdir "%TMPDIR%"
if not exist "wan2gp_core\ckpts" mkdir "wan2gp_core\ckpts"
if not exist "wan2gp_core\loras\minimax_h3" mkdir "wan2gp_core\loras\minimax_h3"
if not exist "outputs" mkdir "outputs"

:: 4. Install PyTorch with CUDA support
echo ==============================================================================
echo [*] Installing PyTorch with CUDA acceleration...
echo ==============================================================================
if "!USE_UV!"=="1" (
    uv pip install --python "%VENV_PY%" torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
) else (
    "!VENV_PY!" -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
)

:: 5. Install Triton & SageAttention for RTX 30/40/50 Series
echo.
echo ==============================================================================
echo [*] Installing Acceleration Kernels (Triton Windows + SageAttention 2)...
echo ==============================================================================
if "!USE_UV!"=="1" (
    uv pip install --python "%VENV_PY%" triton-windows
    uv pip install --python "%VENV_PY%" "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl"
) else (
    "!VENV_PY!" -m pip install triton-windows
    "!VENV_PY!" -m pip install "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl"
)

:: 6. Install Core Requirements
echo.
echo ==============================================================================
echo [*] Installing Application Dependencies...
echo ==============================================================================
if "!USE_UV!"=="1" (
    uv pip install --python "%VENV_PY%" -r wan2gp_core\requirements.txt
    uv pip install --python "%VENV_PY%" tqdm huggingface_hub requests
) else (
    "!VENV_PY!" -m pip install -r wan2gp_core\requirements.txt
    "!VENV_PY!" -m pip install tqdm huggingface_hub requests
)

echo.
echo ==============================================================================
echo [✓] SETUP COMPLETED SUCCESSFULLY!
echo ==============================================================================
echo Next Steps:
echo   1. Run 'download_models.bat' to download MiniMax H3 model weights.
echo   2. Run 'run_windows.bat' to launch the Web UI.
echo ==============================================================================
echo.
pause
