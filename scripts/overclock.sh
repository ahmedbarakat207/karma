#!/usr/bin/env bash
set -e

CONFIG_TXT="/boot/firmware/config.txt"
if [ ! -f "$CONFIG_TXT" ]; then
    CONFIG_TXT="/boot/config.txt"
fi

if [ ! -f "$CONFIG_TXT" ]; then
    echo "Error: config.txt not found at /boot/firmware/config.txt or /boot/config.txt" >&2
    exit 1
fi

MODEL_INFO=""
if [ -f "/proc/device-tree/model" ]; then
    MODEL_INFO=$(tr -d '\0' < /proc/device-tree/model)
fi

TARGET_MODE="${1:-moderate}"

sudo sed -i '/over_voltage/d' "$CONFIG_TXT" 2>/dev/null || true
sudo sed -i '/arm_freq/d' "$CONFIG_TXT" 2>/dev/null || true
sudo sed -i '/gpu_freq/d' "$CONFIG_TXT" 2>/dev/null || true

if [[ "$MODEL_INFO" == *"Raspberry Pi 5"* ]]; then
    if [ "$TARGET_MODE" = "turbo" ] || [ "$TARGET_MODE" = "3000" ]; then
        echo "Applying Raspberry Pi 5 Turbo Overclock (3000 MHz)..."
        echo "arm_freq=3000" | sudo tee -a "$CONFIG_TXT" > /dev/null
    else
        echo "Applying Raspberry Pi 5 Safe Overclock (2800 MHz)..."
        echo "arm_freq=2800" | sudo tee -a "$CONFIG_TXT" > /dev/null
    fi
else
    if [ "$TARGET_MODE" = "turbo" ] || [ "$TARGET_MODE" = "2000" ]; then
        echo "Applying Raspberry Pi 4 Turbo Overclock (2000 MHz, over_voltage=6)..."
        echo "NOTE: Requires high-current 5.1V 3A+ power supply and heatsink."
        echo "arm_freq=2000" | sudo tee -a "$CONFIG_TXT" > /dev/null
        echo "over_voltage=6" | sudo tee -a "$CONFIG_TXT" > /dev/null
    else
        echo "Applying Raspberry Pi 4 Moderate Safe Overclock (1800 MHz, over_voltage=2)..."
        echo "Stable with standard chargers and 20% faster than stock 1500 MHz."
        echo "arm_freq=1800" | sudo tee -a "$CONFIG_TXT" > /dev/null
        echo "over_voltage=2" | sudo tee -a "$CONFIG_TXT" > /dev/null
    fi
fi

sync

echo "Overclock configuration successfully written to $CONFIG_TXT"
echo "To revert back at any time, run: ./scripts/revert_clock.sh"
echo "Reboot to apply: sudo reboot"
