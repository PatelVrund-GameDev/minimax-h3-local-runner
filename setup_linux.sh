#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo "          MINIMAX H3 & WAN2GP - 1-CLICK LINUX SETUP"
echo "=============================================================================="

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "[OK] NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "[!] WARNING: nvidia-smi not found. CUDA acceleration may not be available."
fi

# Clone Wan2GP Core Engine if not present
if [ ! -d "wan2gp_core" ]; then
    echo "[*] Cloning official Wan2GP core engine from upstream..."
    git clone https://github.com/deepbeepmeep/Wan2GP.git wan2gp_core
else
    echo "[OK] Wan2GP core engine directory found."
fi

# Setup Python environment
if command -v uv &> /dev/null; then
    echo "[OK] Using uv package manager."
    if [ ! -d ".venv" ]; then
        uv venv --python 3.11 .venv
    fi
    source .venv/bin/activate
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    uv pip install triton
    uv pip install "setuptools<=75.8.2" ninja wheel
    uv pip install -r wan2gp_core/requirements.txt
    uv pip install tqdm huggingface_hub requests
else
    echo "[*] Setting up python virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install triton
    pip install -r wan2gp_core/requirements.txt
    pip install tqdm huggingface_hub requests
fi

mkdir -p hf_cache torch_cache tmp wan2gp_core/ckpts wan2gp_core/loras/minimax_h3 outputs

echo "=============================================================================="
echo "[OK] Linux setup complete! Run ./run_linux.sh to launch."
echo "=============================================================================="
