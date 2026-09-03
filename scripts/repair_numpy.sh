#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

export PIP_PREFER_BINARY=1
export PIP_ONLY_BINARY="numpy,scipy,spacy,torch,torchvision,torchaudio"
export PIP_EXTRA_INDEX_URL="https://www.piwheels.org/simple"

echo "Uninstalling any broken or conflicting NumPy packages..."
pip uninstall -y numpy 2>/dev/null || true

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
if [ -n "$SITE_PACKAGES" ] && [ -d "$SITE_PACKAGES" ]; then
    rm -rf "$SITE_PACKAGES"/numpy* 2>/dev/null || true
fi

echo "Installing clean prebuilt binary NumPy wheel..."
pip install --force-reinstall --prefer-binary --no-cache-dir numpy

echo "Ensuring sentence-transformers does not break NumPy ABI..."
pip install --prefer-binary --no-deps sentence-transformers 2>/dev/null || true

echo "Validating runtime NumPy sanity check..."
python3 -c "import numpy as np; a = np.arange(10, dtype=np.float32); assert a.sum() == 45.0; print('SUCCESS: NumPy', np.__version__, 'passed sanity check at', np.__file__)"

if systemctl list-units --full -all 2>/dev/null | grep -q "karma.service"; then
    echo "Restarting karma robot service..."
    sudo systemctl restart karma 2>/dev/null || true
fi

echo "NumPy repair complete."
