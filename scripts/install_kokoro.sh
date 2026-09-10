#!/usr/bin/env bash
# Installs Kokoro TTS without hanging at "preparing metadata" on Raspberry Pi.
# Plain `pip install kokoro` builds Rust tokenizers + torch from source (OOM on 4GB).
# This uses piwheels prebuilt binaries, installs deps separately, kokoro with --no-deps.
#
# Usage:
#   bash scripts/install_kokoro.sh            # full: ONNX + PyTorch kokoro
#   bash scripts/install_kokoro.sh --onnx-only # light: skip torch kokoro, use kokoro_q4.onnx
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

ONNX_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --onnx-only) ONNX_ONLY=1 ;;
    esac
done

# Use project venv if present (same as setup.sh / start_robot.sh)
if [ -f "$REPO_DIR/.venv/bin/python3" ]; then
    # shellcheck disable=SC1091
    source "$REPO_DIR/.venv/bin/activate"
fi

export PIP_PREFER_BINARY=1
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://www.piwheels.org/simple}"

echo "[1/4] Upgrading pip tooling..."
python3 -m pip install --prefer-binary --upgrade pip setuptools wheel

echo "[2/4] Checking espeak-ng (required for Kokoro English G2P)..."
if ! command -v espeak-ng >/dev/null 2>&1; then
    echo "  espeak-ng missing, installing via apt..."
    sudo apt-get update -o Acquire::Retries=3 -o Acquire::ForceIPv4=true -y
    sudo apt-get install -y espeak-ng
else
    echo "  espeak-ng OK: $(espeak-ng --version 2>&1 | head -1)"
fi

echo "[3/4] Installing TTS Python deps (prebuilt binaries)..."
python3 -m pip install --prefer-binary --ignore-requires-python \
    loguru transformers "misaki>=0.9.4" kokoro-onnx num2words onnxruntime

if [ "$ONNX_ONLY" -eq 1 ]; then
    echo "  --onnx-only: skipping PyTorch kokoro. Set USE_KOKORO_ONNX=true in .env"
else
    echo "[4/4] Installing kokoro with --no-deps (skips source metadata resolve)..."
    python3 -m pip install --prefer-binary --ignore-requires-python --no-deps kokoro
fi

echo ""
echo "Verifying..."
espeak-ng --version 2>&1 | head -1 || echo "WARN: espeak-ng still missing"
python3 -c "import misaki; print('misaki OK')" 2>&1 | tail -1
if [ "$ONNX_ONLY" -eq 0 ]; then
    python3 -c "from kokoro import KPipeline; print('kokoro OK')" 2>&1 | tail -3
else
    echo "ONNX-only mode: set USE_KOKORO_ONNX=true in .env, models/kokoro_q4.onnx is used"
fi
echo "Done. Re-run: python3 scripts/test_audio.py"
