#!/usr/bin/env bash
set -e

# ==============================================================================
# KARMA DISPLAY & AUDIO STABILIZATION SCRIPT
# Fixes 7-inch LCD HDMI screen blanking / shutting off and on repeatedly,
# and permanently routes audio to the 3.5mm headphone jack or USB DAC.
# ==============================================================================

echo "============================================================"
echo "  Karma Display & Audio Stabilization"
echo "============================================================"

# 1. Locate config.txt
CONFIG_TXT="/boot/firmware/config.txt"
if [ ! -f "$CONFIG_TXT" ]; then
    CONFIG_TXT="/boot/config.txt"
fi

if [ -f "$CONFIG_TXT" ]; then
    echo "[1] Updating $CONFIG_TXT to prevent HDMI video blanking..."

    # Ensure vc4-kms-v3d has noaudio (stops HDMI audio clock resets that blank the screen)
    if grep -q "dtoverlay=vc4-kms-v3d" "$CONFIG_TXT"; then
        if ! grep -q "dtoverlay=vc4-kms-v3d,noaudio" "$CONFIG_TXT"; then
            sudo sed -i 's/dtoverlay=vc4-kms-v3d.*/dtoverlay=vc4-kms-v3d,noaudio/' "$CONFIG_TXT"
            echo "    ✅ Added ',noaudio' to vc4-kms-v3d (disables HDMI audio modeswitch)"
        else
            echo "    ✅ vc4-kms-v3d,noaudio already configured"
        fi
    else
        echo "dtoverlay=vc4-kms-v3d,noaudio" | sudo tee -a "$CONFIG_TXT" >/dev/null
        echo "    ✅ Added dtoverlay=vc4-kms-v3d,noaudio"
    fi

    # Boost HDMI signal strength for 7-inch LCD touchscreens
    if grep -q "^config_hdmi_boost=" "$CONFIG_TXT"; then
        sudo sed -i 's/^config_hdmi_boost=.*/config_hdmi_boost=7/' "$CONFIG_TXT"
    else
        echo "config_hdmi_boost=7" | sudo tee -a "$CONFIG_TXT" >/dev/null
    fi
    echo "    ✅ HDMI signal boost set to 7 (prevents cable signal loss)"

    # Force HDMI hotplug (stops monitor disconnect detection loops)
    if grep -q "^hdmi_force_hotplug=" "$CONFIG_TXT"; then
        sudo sed -i 's/^hdmi_force_hotplug=.*/hdmi_force_hotplug=1/' "$CONFIG_TXT"
    else
        echo "hdmi_force_hotplug=1" | sudo tee -a "$CONFIG_TXT" >/dev/null
    fi
    echo "    ✅ hdmi_force_hotplug=1 enabled"

    # Ensure analog audio is enabled in hardware
    if grep -q "^dtparam=audio=" "$CONFIG_TXT"; then
        sudo sed -i 's/^dtparam=audio=.*/dtparam=audio=on/' "$CONFIG_TXT"
    else
        echo "dtparam=audio=on" | sudo tee -a "$CONFIG_TXT" >/dev/null
    fi
    echo "    ✅ dtparam=audio=on enabled"
else
    echo "⚠️  config.txt not found at /boot/firmware/config.txt or /boot/config.txt"
fi

# 2. Configure PulseAudio & ALSA immediately (no reboot needed for these)
echo ""
echo "[2] Disabling HDMI audio sinks in PulseAudio and maximizing volume..."

# Unmute and max out all ALSA cards
for _ctrl in "Master" "Headphone" "Headphones" "PCM" "Speaker" "Playback"; do
    amixer sset "$_ctrl" 100% unmute 2>/dev/null || true
    for _c in 0 1 2 3 Headphones Device; do
        amixer -c "$_c" sset "$_ctrl" 100% unmute 2>/dev/null || true
    done
done

# Route bcm2835 audio chip to 3.5mm analog jack (numid=3 1)
amixer cset numid=3 1 2>/dev/null || true
amixer -c Headphones cset numid=3 1 2>/dev/null || true

# Suspend all HDMI sinks in PulseAudio and select analog/USB sink
if command -v pactl &>/dev/null; then
    # Suspend any HDMI sink so PulseAudio NEVER sends audio to HDMI
    while read -r _idx _name _rest; do
        if echo "$_name" | grep -qi "hdmi"; then
            pactl suspend-sink "$_name" 1 2>/dev/null || true
            pactl set-sink-mute "$_name" 1 2>/dev/null || true
            echo "    🚫 Suspended HDMI sink: $_name"
        elif echo "$_name" | grep -qiE "usb|uac|dac|speaker|headphone|analog|bcm2835"; then
            pactl suspend-sink "$_name" 0 2>/dev/null || true
            pactl set-default-sink "$_name" 2>/dev/null || true
            pactl set-sink-mute "$_name" 0 2>/dev/null || true
            pactl set-sink-volume "$_name" 100% 2>/dev/null || true
            echo "    🔊 Activated Audio sink: $_name (100% volume)"
        fi
    done < <(pactl list sinks short 2>/dev/null || true)
fi

# PipeWire controls
if command -v wpctl &>/dev/null; then
    wpctl set-mute @DEFAULT_AUDIO_SINK@ 0 2>/dev/null || true
    wpctl set-volume @DEFAULT_AUDIO_SINK@ 1.0 2>/dev/null || true
fi

sudo alsactl store 2>/dev/null || true

# 3. Disable X11 Screen Blanking and DPMS
echo ""
echo "[3] Disabling X11 display sleep and DPMS timeouts..."
if [ -n "$DISPLAY" ]; then
    xset s off 2>/dev/null || true
    xset -dpms 2>/dev/null || true
    xset s noblank 2>/dev/null || true
    echo "    ✅ X11 DPMS and screen blanking turned OFF"
fi

echo ""
echo "============================================================"
echo "  STABILIZATION COMPLETE!"
echo "============================================================"
echo "Next steps:"
echo "1. Restart Karma:"
echo "   sudo systemctl restart karma"
echo "2. If your 7\" screen was previously blanking on boot, a quick"
echo "   reboot ('sudo reboot') will activate the kernel 'noaudio' fix."
echo "============================================================"
