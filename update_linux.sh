#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo "      MINIMAX H3 LOCAL RUNNER & WAN2GP - 1-CLICK UPDATER"
echo "=============================================================================="

echo "[*] Pulling latest repository updates..."
git pull origin main

if [ -d "wan2gp_core" ]; then
    echo "[*] Pulling latest upstream Wan2GP updates..."
    cd wan2gp_core
    git pull origin main
    cd "$SCRIPT_DIR"
fi

if [ -f ".venv/bin/python" ]; then
    echo "[*] Updating Python packages..."
    source .venv/bin/activate
    pip install -r wan2gp_core/requirements.txt -U
fi

echo "=============================================================================="
echo "[OK] Update completed!"
echo "=============================================================================="
