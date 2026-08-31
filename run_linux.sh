#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f ".venv/bin/python" ]; then
    echo "[!] Virtual environment not found. Please run ./setup_linux.sh first."
    exit 1
fi

export HF_HOME="$SCRIPT_DIR/hf_cache"
export TORCH_HOME="$SCRIPT_DIR/torch_cache"
export TMPDIR="$SCRIPT_DIR/tmp"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$SCRIPT_DIR/outputs" "$SCRIPT_DIR/wan2gp_core/ckpts"

source .venv/bin/activate
cd "$SCRIPT_DIR/wan2gp_core"
python wgp.py "$@"
