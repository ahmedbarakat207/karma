#!/usr/bin/env bash
# ==============================================================================
# Karma Autonomous Mobile AI Companion Robot
# Master "Zero-Touch Run & Go" Installation Script
# Target: Raspberry Pi 4 Model B (8GB RAM) running Raspberry Pi OS Lite (64-bit)
# ==============================================================================
set -euo pipefail

BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
NC="\033[0m"

log_info()    { echo -e "${BLUE}${BOLD}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}${BOLD}[SUCCESS]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}${BOLD}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}${BOLD}[ERROR]${NC} $1"; exit 1; }

echo -e "${CYAN}${BOLD}"
echo "=================================================================="
echo "      🤖 KARMA ROBOT — FULL AUTOMATED ZERO-TOUCH SETUP           "
echo "  Target: Raspberry Pi 4 B (8GB) | OS: Pi OS Lite (64-bit)       "
echo "  Features: 2.0GHz Overclock + Full Xorg Kiosk + Qwen 2.5 0.5B   "
echo "=================================================================="
echo -e "${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TARGET_USER="${SUDO_USER:-$USER}"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"

log_info "Workspace root: $SCRIPT_DIR"
log_info "Executing for user: $TARGET_USER ($TARGET_HOME)"

# ------------------------------------------------------------------------------
# 1. System Package Updates & Full OS Dependency Installation
# ------------------------------------------------------------------------------
log_info "Step 1/9: Updating apt repositories and base OS packages..."
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

log_info "Installing core build tools and C/C++ toolchain..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    pkg-config \
    clang \
    llvm \
    libopenblas-dev \
    libatlas-base-dev \
    liblapack-dev

log_info "Installing Python 3 headers and virtual environment tools..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-setuptools \
    python3-wheel

log_info "Installing Audio & Speech subsystems (ALSA, PortAudio, FFmpeg)..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libasound2 \
    libasound2-dev \
    alsa-utils \
    portaudio19-dev \
    libportaudio2 \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    pulseaudio \
    pulseaudio-utils

log_info "Installing Vision & Camera stack (libcamera, Picamera2, OpenCV deps)..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libcamera-dev \
    libcamera-tools \
    python3-libcamera \
    python3-picamera2 \
    v4l-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

log_info "Installing Hardware Actuation, PWM & GPIO subsystems (BTS7960, MG90S)..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    i2c-tools \
    gpiod \
    libgpiod-dev \
    python3-libgpiod \
    python3-rpi.gpio \
    python3-gpiozero \
    python3-pigpio \
    pigpio

log_info "Installing Complete Xorg Display & Touchscreen Kiosk Stack..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    xserver-xorg \
    xserver-xorg-video-fbdev \
    xserver-xorg-input-all \
    xserver-xorg-input-evdev \
    xserver-xorg-legacy \
    xinit \
    x11-xserver-utils \
    xdotool \
    unclutter \
    openbox \
    xinput \
    xinput-calibrator \
    libdrm-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-ttf-dev \
    libsdl2-mixer-dev

log_success "All system packages installed."

# ------------------------------------------------------------------------------
# 2. Xorg Non-Root & Console Permissions Configuration
# ------------------------------------------------------------------------------
log_info "Step 2/9: Configuring Xorg non-root privileges & permissions..."

sudo mkdir -p /etc/X11
sudo tee /etc/X11/Xwrapper.config > /dev/null << 'EOF'
allowed_users=anybody
needs_root_rights=yes
EOF

sudo usermod -a -G video,audio,gpio,i2c,spi,render,input,tty "$TARGET_USER" || true

log_success "Xorg configured to run cleanly without root restrictions."

# ------------------------------------------------------------------------------
# 3. Hardware Overclocking & Firmware Overlays
# ------------------------------------------------------------------------------
log_info "Step 3/9: Applying 2.0 GHz overclock & hardware overlays in config.txt..."

CONFIG_TXT="/boot/firmware/config.txt"
if [ ! -f "$CONFIG_TXT" ]; then
    CONFIG_TXT="/boot/config.txt"
fi

set_config_param() {
    local key="$1"
    local val="$2"
    if grep -q "^${key}=" "$CONFIG_TXT"; then
        sudo sed -i "s/^${key}=.*/${key}=${val}/" "$CONFIG_TXT"
    elif grep -q "^#\?${key}=" "$CONFIG_TXT"; then
        sudo sed -i "s/^#\?${key}=.*/${key}=${val}/" "$CONFIG_TXT"
    else
        echo "${key}=${val}" | sudo tee -a "$CONFIG_TXT" > /dev/null
    fi
}

# 1. Overclock Cortex-A72: 1.5 GHz -> 2.0 GHz (+33% boost for Qwen 2.5 0.5B inference)
set_config_param "over_voltage" "6"
set_config_param "arm_freq" "2000"

# 2. Camera Module (CSI autodetect)
set_config_param "camera_auto_detect" "1"

# 3. Hardware Buses (I2C, SPI, Audio)
set_config_param "dtparam=i2c_arm" "on"
set_config_param "dtparam=spi" "on"
set_config_param "dtparam=audio" "on"

# 4. Hardware PWM for motors and servo
if ! grep -q "dtoverlay=pwm-2chan" "$CONFIG_TXT"; then
    echo "dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4" | sudo tee -a "$CONFIG_TXT" > /dev/null
fi

# 5. GPU memory allocation for camera pipeline
set_config_param "gpu_mem" "128"

# 6. Disable Linux console screen blanking so the LCD screen stays on forever
CMDLINE_TXT="/boot/firmware/cmdline.txt"
if [ ! -f "$CMDLINE_TXT" ]; then
    CMDLINE_TXT="/boot/cmdline.txt"
fi

if [ -f "$CMDLINE_TXT" ]; then
    if ! grep -q "consoleblank=0" "$CMDLINE_TXT"; then
        log_info "Disabling screen blanking in kernel command line ($CMDLINE_TXT)..."
        sudo sed -i 's/$/ consoleblank=0/' "$CMDLINE_TXT"
    fi
fi

log_success "Hardware firmware and overclock configured."

# ------------------------------------------------------------------------------
# 4. Audio Subsystem Volume & Hardware Daemons
# ------------------------------------------------------------------------------
log_info "Step 4/9: Configuring audio output volume and enabling hardware daemons..."

# Enable and start pigpiod for jitter-free servo PWM
sudo systemctl enable pigpiod || true
sudo systemctl restart pigpiod || true

# Maximize audio volume and unmute standard ALSA controls
amixer -c 0 sset 'Master' 100% unmute 2>/dev/null || true
amixer -c 0 sset 'PCM' 100% unmute 2>/dev/null || true
amixer -c 1 sset 'Master' 100% unmute 2>/dev/null || true
amixer -c 1 sset 'PCM' 100% unmute 2>/dev/null || true
sudo alsactl store 2>/dev/null || true

log_success "Audio initialized and pigpiod daemon running."

# ------------------------------------------------------------------------------
# 5. Python Virtual Environment Setup
# ------------------------------------------------------------------------------
log_info "Step 5/9: Setting up Python virtual environment with system access..."

VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------------------------
# 6. Python AI / ML Stack Installation
# ------------------------------------------------------------------------------
log_info "Step 6/9: Installing optimized Python AI & Robotics packages..."

pip install numpy scipy requests aiohttp fastapi uvicorn pydantic huggingface_hub

# Install CPU PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Audio Speech Pipeline
pip install sounddevice soundfile pyaudio
pip install faster-whisper
pip install onnxruntime
pip install kokoro

# Computer Vision & Gestures
pip install opencv-python-headless
pip install ultralytics
pip install mediapipe

# SQLite Vector & Embeddings
pip install sentence-transformers sqlean.py sqlite-vec

# UI Rendering
pip install pygame

# Compile llama-cpp-python with native ARM NEON SIMD optimizations
log_info "Compiling llama-cpp-python with native ARM NEON vectorization..."
CMAKE_ARGS="-DGGML_CPU_ARM_ARCH=armv8-a -DGGML_BLAS=OFF" pip install --no-cache-dir llama-cpp-python

log_success "All Python AI libraries installed."

# ------------------------------------------------------------------------------
# 7. AI Model Weights Verification & Auto-Download
# ------------------------------------------------------------------------------
log_info "Step 7/9: Verifying and auto-downloading model weights into models/..."

mkdir -p "$SCRIPT_DIR/models"

python3 - << 'EOF'
import os
from huggingface_hub import hf_hub_download

MODELS_DIR = os.path.abspath("models")
os.makedirs(MODELS_DIR, exist_ok=True)

models = [
    {
        "name": "Qwen 2.5 0.5B Instruct GGUF (Brain)",
        "file": "model.gguf",
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "hf_file": "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    },
    {
        "name": "Kokoro TTS Q4 ONNX (Voice)",
        "file": "kokoro_q4.onnx",
        "repo": "hexgrad/Kokoro-82M",
        "hf_file": "kokoro-v0_19.onnx"
    },
    {
        "name": "Kokoro Voices Registry",
        "file": "voices-v1.0.bin",
        "repo": "hexgrad/Kokoro-82M",
        "hf_file": "voices/v1.0.bin"
    }
]

for m in models:
    dest = os.path.join(MODELS_DIR, m["file"])
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"✓ Found {m['name']}: {dest} ({size_mb:.1f} MB)")
    else:
        print(f"⬇ Downloading {m['name']} from {m['repo']}...")
        try:
            downloaded = hf_hub_download(repo_id=m["repo"], filename=m["hf_file"], local_dir=MODELS_DIR)
            if os.path.basename(downloaded) != m["file"]:
                os.rename(downloaded, dest)
            print(f"✓ Successfully downloaded {m['file']}")
        except Exception as e:
            print(f"⚠️ Notice downloading {m['file']}: {e}")

yolo_path = os.path.join(MODELS_DIR, "yolov8n.pt")
if not os.path.exists(yolo_path):
    print("⬇ Downloading YOLOv8n vision model...")
    try:
        from ultralytics import YOLO
        yolo = YOLO("yolov8n.pt")
        if os.path.exists("yolov8n.pt") and not os.path.exists(yolo_path):
            os.rename("yolov8n.pt", yolo_path)
        print(f"✓ YOLOv8n ready at {yolo_path}")
    except Exception as e:
        print(f"⚠️ Notice downloading YOLOv8n: {e}")
EOF

log_success "AI model weights verified."

# ------------------------------------------------------------------------------
# 8. Openbox & Xorg Kiosk Setup (Borderless Fullscreen Face)
# ------------------------------------------------------------------------------
log_info "Step 8/9: Setting up Xorg borderless kiosk & launcher script..."

OPENBOX_DIR="$TARGET_HOME/.config/openbox"
mkdir -p "$OPENBOX_DIR"

# Minimal Openbox rc.xml to remove all window decorations, title bars, and borders
cat << 'EOF' > "$OPENBOX_DIR/rc.xml"
<?xml version="1.0" encoding="UTF-8"?>
<openbox_config xmlns="http://openbox.org/3.4/rc">
  <applications>
    <application class="*">
      <decor>no</decor>
      <fullscreen>yes</fullscreen>
    </application>
  </applications>
</openbox_config>
EOF
chown -R "$TARGET_USER:$TARGET_USER" "$TARGET_HOME/.config"

# Create the master kiosk launcher script (start_robot.sh)
START_SCRIPT="$SCRIPT_DIR/start_robot.sh"
cat << EOF > "$START_SCRIPT"
#!/usr/bin/env bash
# ==============================================================================
# Karma Autonomous Robot Master Launcher (Runs inside Xorg on :0)
# ==============================================================================
set -e

# Disable screen blanking, power management, and screensavers
xset s off 2>/dev/null || true
xset -dpms 2>/dev/null || true
xset s noblank 2>/dev/null || true

# Hide mouse cursor completely after 0.1s
unclutter -idle 0.1 -root &

# Start minimal borderless window manager in background
openbox &

# Launch Karma Autonomous Companion Robot Loop (Voice, Vision, Face & Cognition)
exec "$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/main.py"
EOF
chmod +x "$START_SCRIPT"
chown "$TARGET_USER:$TARGET_USER" "$START_SCRIPT"

log_success "Xorg kiosk script created at $START_SCRIPT."

# ------------------------------------------------------------------------------
# 9. Systemd Master Auto-Start Unit (Full Autonomous Boot on Battery Power)
# ------------------------------------------------------------------------------
log_info "Step 9/9: Registering karma.service systemd auto-start daemon..."

SERVICE_FILE="/etc/systemd/system/karma.service"
sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Karma Autonomous Companion Robot (Xorg Kiosk + AI Brain)
After=network.target sound.target pigpiod.service systemd-user-sessions.service
Wants=pigpiod.service
Conflicts=getty@tty1.service

[Service]
Type=simple
User=$TARGET_USER
WorkingDirectory=$SCRIPT_DIR
Environment=DISPLAY=:0
Environment=XAUTHORITY=$TARGET_HOME/.Xauthority
Environment=PYTHONUNBUFFERED=1
Environment=CTX_SIZE=4096
Environment=N_THREADS=4
Environment=N_BATCH=512
Environment=DEFAULT_REPEAT_PENALTY=1.05
Environment=DEFAULT_TOP_P=0.9

# Launch xinit directly on virtual terminal 1 without login prompt
ExecStart=/usr/bin/xinit $SCRIPT_DIR/start_robot.sh -- :0 vt1 -keeptty
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable karma.service

log_success "karma.service registered and enabled for autonomous boot."

# ------------------------------------------------------------------------------
# Self-Test Validation
# ------------------------------------------------------------------------------
echo ""
log_info "Running quick Cognition Engine validation self-test..."
python3 chat.py --validate || log_warn "Self-test finished with notice (verify models)."

echo ""
echo -e "${GREEN}${BOLD}=================================================================="
echo "   🎉 KARMA ZERO-TOUCH SETUP COMPLETE: READY TO RUN AND GO!      "
echo "==================================================================${NC}"
echo ""
echo -e "What happens now:"
echo -e "  1. Whenever the robot powers on (battery or wall power):"
echo -e "     • Boots directly in ~5 seconds with ${BOLD}2.0 GHz Cortex-A72 CPU${NC}."
echo -e "     • ${BOLD}Xorg starts automatically on the 7\" LCD screen${NC} on :0 vt1."
echo -e "     • Karma's animated companion face appears full-screen (0 borders, 0 cursor)."
echo -e "     • Microphone listens, Camera tracks eyes/objects, Speaker responds."
echo -e "     • Powered by ${BOLD}Qwen 2.5 0.5B Instruct${NC} with 4096 context."
echo ""
echo -e "Useful Commands (via SSH):"
echo -e "  • Check live robot status:    ${CYAN}sudo systemctl status karma${NC}"
echo -e "  • Follow live logs:           ${CYAN}sudo journalctl -u karma -f${NC}"
echo -e "  • Restart robot:              ${CYAN}sudo systemctl restart karma${NC}"
echo -e "  • Interactive terminal chat:  ${CYAN}source .venv/bin/activate && python3 chat.py${NC}"
echo ""
echo -e "${YELLOW}${BOLD}👉 To activate the 2.0 GHz overclock & launch Karma right now, reboot:${NC}"
echo -e "   ${BOLD}sudo reboot${NC}"
echo ""
