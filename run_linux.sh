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

echo "=============================================================================="
echo "          MINIMAX H3 & WAN2GP - LOCAL STUDIO RUNNER"
echo "=============================================================================="
echo "[*] Initializing Wan2GP engine & GPU acceleration kernels..."
echo "[*] Web server starting up at http://127.0.0.1:7860"
echo "=============================================================================="

# Launch background watcher to automatically open browser when port 7860 is ready
(
    for i in {1..120}; do
        if nc -z 127.0.0.1 7860 2>/dev/null || timeout 1 bash -c '</dev/tcp/127.0.0.1/7860' 2>/dev/null; then
            if command -v xdg-open &> /dev/null; then
                xdg-open "http://127.0.0.1:7860" &> /dev/null &
            elif command -v sensible-browser &> /dev/null; then
                sensible-browser "http://127.0.0.1:7860" &> /dev/null &
            fi
            break
        fi
        sleep 1
    done
) &

source .venv/bin/activate
cd "$SCRIPT_DIR/wan2gp_core"
python wgp.py "$@"
