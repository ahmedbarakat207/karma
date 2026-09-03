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

sudo sed -i '/over_voltage/d' "$CONFIG_TXT" 2>/dev/null || true
sudo sed -i '/arm_freq/d' "$CONFIG_TXT" 2>/dev/null || true
sudo sed -i '/gpu_freq/d' "$CONFIG_TXT" 2>/dev/null || true
sudo sed -i '/core_freq/d' "$CONFIG_TXT" 2>/dev/null || true
sudo sed -i '/force_turbo/d' "$CONFIG_TXT" 2>/dev/null || true

sync

echo "CPU frequency and core voltage successfully reverted to stock defaults in $CONFIG_TXT"
echo "Reboot the system to apply: sudo reboot"
