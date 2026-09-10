#!/usr/bin/env bash
# Installs ALL Karma Python deps as prebuilt binaries on Raspberry Pi — never builds.
# Any package with no binary wheel is SKIPPED with a message instead of hanging
# in a source build (the `pip install kokoro` / dlib / tokenizers trap).
#
# Usage:  bash scripts/install_all_bins.sh 2>&1 | tee /tmp/bins_install.txt
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if [ -f "$REPO_DIR/.venv/bin/python3" ]; then
    # shellcheck disable=SC1091
    source "$REPO_DIR/.venv/bin/activate"
fi

# Binary-only policy: pip FAILS instead of compiling from source.
export PIP_PREFER_BINARY=1
export PIP_ONLY_BINARY="numpy,scipy,spacy,torch,torchvision,torchaudio"
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://www.piwheels.org/simple}"

PASS=0; SKIP=0; FAILED=()
bin_install() {
    local label="$1"; shift
    echo "--- $label ---"
    # --only-binary=:all: is the hard guarantee: no compiler ever runs.
    if python3 -m pip install --only-binary=:all: --prefer-binary "$@" 2>&1 | tail -2; then
        PASS=$((PASS + 1))
    else
        echo "  SKIP: no binary wheel for: $* (not building from source)"
        SKIP=$((SKIP + 1)); FAILED+=("$label")
    fi
}

echo "[0] pip tooling..."
python3 -m pip install --only-binary=:all: --prefer-binary --upgrade pip setuptools wheel
python3 -c "import numpy" 2>/dev/null || \
    python3 -m pip install --only-binary=:all: --prefer-binary --force-reinstall numpy

bin_install "web/server" requests aiohttp fastapi uvicorn pydantic huggingface_hub groq
bin_install "torch-cpu" torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
bin_install "audio" sounddevice soundfile faster-whisper onnxruntime silero-vad
bin_install "tts-deps" --ignore-requires-python loguru transformers "misaki>=0.9.4" kokoro-onnx num2words
bin_install "kokoro(no-deps)" --ignore-requires-python --no-deps kokoro
bin_install "vision" --no-build-isolation ultralytics mediapipe
bin_install "memory/rag" sqlean.py sqlite-vec markitdown pdfminer.six
bin_install "memory/embeddings(no-deps)" --no-deps sentence-transformers
bin_install "hardware" pygame pigpio gpiozero
bin_install "llm" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu llama-cpp-python

echo ""
echo "NOTE: opencv + face_recognition/dlib have no Pi wheels and are"
echo "deliberately NOT built here (dlib compile takes hours, OOMs on 4GB)."
echo "Use apt system packages with a --system-site-packages venv instead:"
echo "  sudo apt install -y python3-opencv"
echo "  python3 -m venv --system-site-packages .venv  (or reuse setup.sh)"

echo ""
echo "=== VERIFY (import smoke test) ==="
for mod in numpy torch sounddevice faster_whisper onnxruntime kokoro misaki sentence_transformers ultralytics mediapipe llama_cpp markitdown groq fastapi cv2; do
    if python3 -c "import $mod" 2>/dev/null; then echo "  OK   $mod"; else echo "  MISS $mod"; fi
done

echo ""
echo "=== SUMMARY: $PASS groups installed, $SKIP skipped ==="
if [ "${#FAILED[@]}" -gt 0 ]; then echo "skipped: ${FAILED[*]}"; fi
