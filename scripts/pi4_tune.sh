#!/usr/bin/env bash
# ============================================================
# Karma — Raspberry Pi 4 one-shot OS tuning script
# Run once on the Pi:  sudo bash scripts/pi4_tune.sh
# ============================================================
set -euo pipefail

echo "=== Karma Pi 4 OS Tuner ==="

# ── 1. gpu_mem ────────────────────────────────────────────────
CONFIG_TXT="/boot/config.txt"
if grep -q "^gpu_mem=" "$CONFIG_TXT" 2>/dev/null; then
    sudo sed -i 's/^gpu_mem=.*/gpu_mem=64/' "$CONFIG_TXT"
else
    echo "gpu_mem=64" | sudo tee -a "$CONFIG_TXT" >/dev/null
fi
echo "  ✓ gpu_mem=64  (freed ~64 MB from GPU pool)"

# ── 2. Disable unused services ────────────────────────────────
SERVICES=(bluetooth avahi-daemon cups ModemManager)
for svc in "${SERVICES[@]}"; do
    if systemctl list-unit-files "${svc}.service" &>/dev/null; then
        sudo systemctl disable --now "${svc}.service" 2>/dev/null || true
        echo "  ✓ disabled ${svc}"
    fi
done

# ── 3. Overclock to moderate (1800 MHz) ──────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/overclock.sh" ]; then
    bash "$SCRIPT_DIR/overclock.sh" moderate
    echo "  ✓ overclock set to moderate (1800 MHz)"
else
    echo "  ⚠  scripts/overclock.sh not found — skipping overclock"
fi

echo ""
echo "=== Done. Reboot for gpu_mem + overclock to take effect. ==="
echo "    sudo reboot"
