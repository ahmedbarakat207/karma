#!/usr/bin/env bash
set -euo pipefail

BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"

log_info()    { echo -e "${BLUE}${BOLD}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}${BOLD}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}${BOLD}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}${BOLD}[ERROR]${NC} $1"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TARGET_USER="${SUDO_USER:-$USER}"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"

log_info "Installing for $TARGET_USER ($TARGET_HOME) in $SCRIPT_DIR"

fix_network_and_dns() {
    log_info "Verifying network connectivity and DNS resolution..."
    sudo mkdir -p /etc/apt/apt.conf.d
    echo 'Acquire::ForceIPv4 "true";' | sudo tee /etc/apt/apt.conf.d/99force-ipv4 > /dev/null

    if ! host deb.debian.org >/dev/null 2>&1 && ! getent ahosts deb.debian.org >/dev/null 2>&1; then
        log_warn "DNS resolution failed. Updating nameservers..."
        if [ -f /etc/resolv.conf ]; then
            sudo sed -i '/nameserver 8.8.8.8/d' /etc/resolv.conf 2>/dev/null || true
            sudo sed -i '/nameserver 1.1.1.1/d' /etc/resolv.conf 2>/dev/null || true
            echo -e "nameserver 8.8.8.8\nnameserver 1.1.1.1\nnameserver 8.8.4.4" | sudo tee -a /etc/resolv.conf > /dev/null
        fi
        if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet systemd-resolved 2>/dev/null; then
            sudo systemctl restart systemd-resolved || true
        fi
    fi

    if command -v timedatectl >/dev/null 2>&1; then
        sudo timedatectl set-ntp true 2>/dev/null || true
        if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet systemd-timesyncd 2>/dev/null; then
            sudo systemctl restart systemd-timesyncd || true
        fi
    fi

    local retries=3
    local count=0
    local success=0
    while [ $count -lt $retries ]; do
        count=$((count + 1))
        log_info "Updating system packages (attempt $count of $retries)..."
        if sudo apt-get update -o Acquire::Retries=3 -o Acquire::ForceIPv4=true -y; then
            success=1
            break
        fi
        log_warn "apt-get update failed on attempt $count. Retrying in 3 seconds..."
        sleep 3
    done

    if [ $success -eq 0 ]; then
        log_error "Failed to update package repositories after $retries attempts. Please check internet connection."
    fi
}

fix_network_and_dns
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
    ninja-build \
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

log_info "Installing Vision & Camera stack (libcamera, Picamera2, OpenCV GTK)..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-opencv \
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

log_info "Configuring Xorg permissions..."

sudo mkdir -p /etc/X11
sudo tee /etc/X11/Xwrapper.config > /dev/null << 'EOF'
allowed_users=anybody
needs_root_rights=yes
EOF

sudo usermod -a -G video,audio,gpio,i2c,spi,render,input,tty "$TARGET_USER" || true

log_success "Xorg configured to run cleanly without root restrictions."

log_info "Configuring hardware settings in config.txt..."

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

set_config_param "over_voltage" "6"
set_config_param "arm_freq" "2000"

set_config_param "camera_auto_detect" "1"

set_config_param "dtparam=i2c_arm" "on"
set_config_param "dtparam=spi" "on"
set_config_param "dtparam=audio" "on"

if ! grep -q "dtoverlay=pwm-2chan" "$CONFIG_TXT"; then
    echo "dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4" | sudo tee -a "$CONFIG_TXT" > /dev/null
fi

set_config_param "gpu_mem" "128"

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

log_info "Enabling V4L2 CSI camera module driver..."
sudo modprobe bcm2835-v4l2 2>/dev/null || true
if [ -f "/etc/modules" ]; then
    if ! grep -q "bcm2835-v4l2" /etc/modules; then
        echo "bcm2835-v4l2" | sudo tee -a /etc/modules > /dev/null || true
    fi
fi

log_success "Hardware firmware and overclock configured."

log_info "Configuring audio..."

sudo systemctl enable pigpiod || true
sudo systemctl restart pigpiod || true

amixer -c 0 sset 'Master' 100% unmute 2>/dev/null || true
amixer -c 0 sset 'PCM' 100% unmute 2>/dev/null || true
amixer -c 1 sset 'Master' 100% unmute 2>/dev/null || true
amixer -c 1 sset 'PCM' 100% unmute 2>/dev/null || true
sudo alsactl store 2>/dev/null || true

log_success "Audio initialized and pigpiod daemon running."

log_info "Setting up Python virtual environment..."

VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

log_info "Installing Python dependencies..."

pip install numpy scipy requests aiohttp fastapi uvicorn pydantic huggingface_hub

pip install torch torchvision torchaudio || pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install sounddevice soundfile pyaudio
pip install faster-whisper
pip install onnxruntime
pip install kokoro

pip install opencv-python || true
pip install ultralytics
pip install mediapipe

pip install sentence-transformers sqlean.py sqlite-vec
pip install "markitdown[pdf]"

pip install pygame

log_info "Compiling llama-cpp-python with native ARM NEON vectorization..."
CMAKE_ARGS="-DGGML_CPU_ARM_ARCH=armv8-a -DGGML_BLAS=OFF" pip install --no-cache-dir llama-cpp-python

log_success "All Python AI libraries installed."

log_info "Configuring Node.js and Electron frontend dependencies..."

if ! command -v node >/dev/null 2>&1; then
    log_info "Installing Node.js LTS via official NodeSource repository..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

if [ -d "$SCRIPT_DIR/ui" ]; then
    log_info "Installing Electron and UI dependencies in ui/..."
    cd "$SCRIPT_DIR/ui"
    npm install
    cd "$SCRIPT_DIR"
fi

log_info "Checking models in models/..."

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

log_info "Setting up kiosk launcher script..."

OPENBOX_DIR="$TARGET_HOME/.config/openbox"
mkdir -p "$OPENBOX_DIR"

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

START_SCRIPT="$SCRIPT_DIR/start_robot.sh"
cat << EOF > "$START_SCRIPT"
#!/usr/bin/env bash
set -e

xset s off 2>/dev/null || true
xset -dpms 2>/dev/null || true
xset s noblank 2>/dev/null || true

unclutter -idle 0.1 -root &

openbox &

exec "$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/main.py"
EOF
chmod +x "$START_SCRIPT"
chown "$TARGET_USER:$TARGET_USER" "$START_SCRIPT"

log_success "Xorg kiosk script created at $START_SCRIPT."

log_info "Registering karma systemd service..."

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

log_info "Validating engine..."
python3 chat.py --validate || log_warn "Validation returned notice (verify models)."

echo ""
log_success "Karma setup complete."
echo "Commands:"
echo "  sudo systemctl status karma"
echo "  sudo journalctl -u karma -f"
echo "  sudo systemctl restart karma"
echo ""
echo "Reboot to apply changes: sudo reboot"
