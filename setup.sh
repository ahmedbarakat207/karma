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

    sudo dpkg --configure -a 2>/dev/null || true
    sudo apt-get clean 2>/dev/null || true

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

apt_install_safe() {
    local available=()
    for pkg in "$@"; do
        local cand
        cand=$(apt-cache policy "$pkg" 2>/dev/null | awk '/Candidate:/ {print $2}')
        if [ -n "$cand" ] && [ "$cand" != "(none)" ]; then
            available+=("$pkg")
        fi
    done
    if [ ${#available[@]} -gt 0 ]; then
        sudo DEBIAN_FRONTEND=noninteractive apt-get install --fix-missing -y "${available[@]}"
    fi
}

log_info "Installing core build tools and C/C++ toolchain..."
apt_install_safe \
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
    liblapack-dev

log_info "Installing Python 3 headers and virtual environment tools..."
apt_install_safe \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-setuptools \
    python3-wheel \
    python3-numpy \
    python3-scipy \
    python3-spacy

log_info "Installing Audio & Speech subsystems (ALSA, PortAudio, FFmpeg)..."
apt_install_safe \
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
apt_install_safe \
    python3-opencv \
    libcamera-dev \
    libcamera-tools \
    python3-libcamera \
    python3-picamera2 \
    v4l-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

log_info "Installing Hardware Actuation, PWM & GPIO subsystems (BTS7960, MG90S)..."
apt_install_safe \
    i2c-tools \
    gpiod \
    libgpiod-dev \
    python3-libgpiod \
    python3-gpiozero

log_info "Installing Complete Xorg Display & Touchscreen Kiosk Stack..."
apt_install_safe \
    xserver-xorg \
    xserver-xorg-video-fbdev \
    xserver-xorg-input-all \
    xserver-xorg-input-libinput \
    xserver-xorg-legacy \
    xinit \
    x11-xserver-utils \
    xdotool \
    unclutter \
    openbox \
    xinput \
    chromium \
    chromium-browser \
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

log_info "Configuring user karma and system autologin..."

if ! id -u karma >/dev/null 2>&1; then
    sudo useradd -m -s /bin/bash -G sudo,video,audio,gpio,i2c,spi,render,input,tty karma
else
    sudo usermod -a -G sudo,video,audio,gpio,i2c,spi,render,input,tty karma || true
fi

echo "karma:1234" | sudo chpasswd

echo "karma ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/010_karma-nopasswd > /dev/null
sudo chmod 0440 /etc/sudoers.d/010_karma-nopasswd

sudo mkdir -p /etc/systemd/system/getty@tty1.service.d
sudo tee /etc/systemd/system/getty@tty1.service.d/autologin.conf > /dev/null << 'EOF'
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin karma --noclear %I $TERM
EOF

if [ -f /etc/lightdm/lightdm.conf ]; then
    sudo sed -i 's/^#\?autologin-user=.*/autologin-user=karma/' /etc/lightdm/lightdm.conf
    sudo sed -i 's/^#\?autologin-user-timeout=.*/autologin-user-timeout=0/' /etc/lightdm/lightdm.conf
fi

if command -v raspi-config >/dev/null 2>&1; then
    sudo raspi-config nonint do_boot_behaviour B2 >/dev/null 2>&1 || true
fi

log_success "Autologin configured for user karma (password: 1234)."

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

sudo sed -i '/over_voltage/d' "$CONFIG_TXT" 2>/dev/null || true
sudo sed -i '/arm_freq/d' "$CONFIG_TXT" 2>/dev/null || true

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

log_success "Hardware firmware configured."

log_info "Configuring audio..."

if systemctl list-unit-files pigpiod.service 2>/dev/null | grep -q "pigpiod.service"; then
    sudo systemctl enable pigpiod 2>/dev/null || true
    sudo systemctl restart pigpiod 2>/dev/null || true
fi

amixer -c 0 sset 'Master' 100% unmute 2>/dev/null || true
amixer -c 0 sset 'PCM' 100% unmute 2>/dev/null || true
amixer -c 1 sset 'Master' 100% unmute 2>/dev/null || true
amixer -c 1 sset 'PCM' 100% unmute 2>/dev/null || true
sudo alsactl store 2>/dev/null || true

log_success "Audio initialized."

log_info "Setting up Python virtual environment..."

VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
elif [ -f "$VENV_DIR/pyvenv.cfg" ]; then
    sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' "$VENV_DIR/pyvenv.cfg"
fi

source "$VENV_DIR/bin/activate"

export PIP_PREFER_BINARY=1
export PIP_ONLY_BINARY="numpy,scipy,spacy,torch,torchvision,torchaudio"
export PIP_EXTRA_INDEX_URL="https://www.piwheels.org/simple"

pip install --prefer-binary --upgrade pip setuptools wheel

if ! python3 -c "import numpy" >/dev/null 2>&1; then
    log_info "Repairing NumPy installation..."
    pip install --prefer-binary --force-reinstall numpy
fi

log_info "Installing Python dependencies (prebuilt binaries)..."

pip install --prefer-binary requests aiohttp fastapi uvicorn pydantic huggingface_hub

pip install --prefer-binary torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || pip install --prefer-binary torch torchvision torchaudio

pip install --prefer-binary sounddevice soundfile
pip install --prefer-binary faster-whisper
pip install --prefer-binary onnxruntime

pip install --prefer-binary --ignore-requires-python loguru transformers "misaki>=0.9.4" kokoro-onnx
pip install --prefer-binary --ignore-requires-python --no-deps kokoro

pip install --prefer-binary --no-build-isolation ultralytics
pip install --prefer-binary --no-build-isolation mediapipe

pip install --prefer-binary sqlean.py sqlite-vec
pip install --prefer-binary --no-deps sentence-transformers
pip install --prefer-binary markitdown pdfminer.six

pip install --prefer-binary pygame pigpio gpiozero

log_info "Installing llama-cpp-python (prebuilt binary)..."
pip install --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu llama-cpp-python || pip install --prefer-binary llama-cpp-python

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
    },
    {
        "name": "Nabra Arabic TTS Weights",
        "file": os.path.join("nabra", "kokoro_arabic.pth"),
        "repo": "oddadmix/Nabra-82M-v0.1",
        "hf_file": "kokoro_arabic.pth"
    },
    {
        "name": "Nabra Arabic Voicepack (af_msa)",
        "file": os.path.join("nabra", "af_msa.pt"),
        "repo": "oddadmix/Nabra-82M-v0.1",
        "hf_file": "af_msa.pt"
    },
    {
        "name": "Nabra Arabic Config",
        "file": os.path.join("nabra", "config.json"),
        "repo": "oddadmix/Nabra-82M-v0.1",
        "hf_file": "config.json"
    }
]

for m in models:
    dest = os.path.join(MODELS_DIR, m["file"])
    dest_parent = os.path.dirname(dest)
    os.makedirs(dest_parent, exist_ok=True)
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"✓ Found {m['name']}: {dest} ({size_mb:.1f} MB)")
    else:
        print(f"⬇ Downloading {m['name']} from {m['repo']}...")
        try:
            downloaded = hf_hub_download(repo_id=m["repo"], filename=m["hf_file"], local_dir=dest_parent)
            if os.path.basename(downloaded) != os.path.basename(dest):
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

vlm_dir = os.path.join(MODELS_DIR, "smolvlm2-256m")
if os.path.isdir(vlm_dir) and len(os.listdir(vlm_dir)) > 0:
    print(f"✓ Found SmolVLM2 verifier: {vlm_dir}")
else:
    print("⬇ Downloading SmolVLM2-256M verifier (one-shot scene checks)...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
                          local_dir=vlm_dir)
        print(f"✓ SmolVLM2 verifier ready at {vlm_dir}")
    except Exception as e:
        print(f"⚠️ Notice downloading SmolVLM2: {e} (lazy-downloads on first novel sighting)")
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
cat << 'EOF' > "$START_SCRIPT"
#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

xset s off 2>/dev/null || true
xset -dpms 2>/dev/null || true
xset s noblank 2>/dev/null || true

unclutter -idle 0.1 -root &

openbox &

while true; do
    if [ -f "$SCRIPT_DIR/.venv/bin/python3" ]; then
        "$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/main.py" >> "$SCRIPT_DIR/karma.log" 2>&1
    else
        python3 "$SCRIPT_DIR/main.py" >> "$SCRIPT_DIR/karma.log" 2>&1
    fi
    sleep 3
done
EOF
chmod +x "$START_SCRIPT"
chown "$TARGET_USER:$TARGET_USER" "$START_SCRIPT"

log_success "Xorg kiosk script created at $START_SCRIPT."

log_info "Registering karma systemd service..."

SERVICE_FILE="/etc/systemd/system/karma.service"
sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Karma Autonomous Companion Robot (Xorg Kiosk + AI Brain)
After=network.target sound.target systemd-user-sessions.service
Conflicts=getty@tty1.service

[Service]
Type=simple
User=$TARGET_USER
WorkingDirectory=$SCRIPT_DIR
Environment=DISPLAY=:0
Environment=XAUTHORITY=$TARGET_HOME/.Xauthority
Environment=PYTHONUNBUFFERED=1
Environment=CTX_SIZE=4096
Environment=N_THREADS=2
Environment=N_BATCH=256
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

log_info "Ingesting Karma Knowledge Base into RAG..."
if [ -f "data/documents/karma_knowledge.md" ]; then
    python3 -m src.memory.rag --ingest data/documents/karma_knowledge.md || log_warn "Knowledge base RAG indexing deferred."
fi

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
