#!/usr/bin/env bash
# =============================================================================
# Karma — Pi-side patch script
# Run on the robot after any git push to apply updates without a full re-setup.
#
#   bash scripts/patch.sh               # git pull + patch service + restart
#   bash scripts/patch.sh --no-restart  # patch only, don't restart yet
#   bash scripts/patch.sh --restart-only # skip pull/patch, just restart
# =============================================================================

set -euo pipefail

BOLD="\033[1m"; GREEN="\033[0;32m"; BLUE="\033[0;34m"
YELLOW="\033[1;33m"; RED="\033[0;31m"; NC="\033[0m"
ok()   { echo -e "${GREEN}${BOLD}  ✓${NC}  $1"; }
info() { echo -e "${BLUE}${BOLD}  →${NC}  $1"; }
warn() { echo -e "${YELLOW}${BOLD}  ⚠${NC}  $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICE_FILE="/etc/systemd/system/karma.service"
ENV_FILE="$REPO_DIR/.env"

ARG="${1:-}"
DO_PULL=true
DO_PATCH=true
DO_RESTART=true
[[ "$ARG" == "--no-restart"   ]] && DO_RESTART=false
[[ "$ARG" == "--restart-only" ]] && DO_PULL=false && DO_PATCH=false

echo ""
echo -e "${BOLD}╔══════════════════════════════════════╗${NC}"
echo -e "${BOLD}║       KARMA  PATCH  SCRIPT           ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════╝${NC}"
echo ""

# ── 1. Git pull ───────────────────────────────────────────────────────────────
if $DO_PULL; then
    info "Pulling latest code from origin/main…"
    cd "$REPO_DIR"
    git fetch origin
    BEHIND=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
    if [ "$BEHIND" -gt 0 ]; then
        git pull --ff-only origin main
        ok "Pulled $BEHIND new commit(s)"
    else
        ok "Already up to date"
    fi
    chmod +x "$REPO_DIR/start_robot.sh" "$REPO_DIR/scripts/"*.sh 2>/dev/null || true
fi

# ── 2. Patch karma.service ────────────────────────────────────────────────────
if $DO_PATCH; then
    if [ ! -f "$SERVICE_FILE" ]; then
        warn "karma.service not found at $SERVICE_FILE — skipping service patch"
        warn "Run setup.sh first, then re-run this script."
    else
        info "Patching $SERVICE_FILE…"

        # 2a. Remove stale hardcoded env vars that override .env
        for v in CTX_SIZE N_THREADS N_BATCH DEFAULT_REPEAT_PENALTY DEFAULT_TOP_P; do
            if grep -q "^Environment=${v}=" "$SERVICE_FILE" 2>/dev/null; then
                sudo sed -i "/^Environment=${v}=/d" "$SERVICE_FILE"
                ok "Removed hardcoded Environment=${v}="
            fi
        done

        # 2b. Add EnvironmentFile if not already there
        ENV_LINE="EnvironmentFile=-${ENV_FILE}"
        if ! grep -qF "$ENV_LINE" "$SERVICE_FILE"; then
            sudo sed -i "/^WorkingDirectory=/a ${ENV_LINE}" "$SERVICE_FILE"
            ok "Added EnvironmentFile=-$ENV_FILE"
        else
            ok "EnvironmentFile already present — no change"
        fi

        # 2c. Ensure HOME is set (needed for .Xauthority lookup)
        TARGET_USER="$(stat -c '%U' "$REPO_DIR" 2>/dev/null || echo "$USER")"
        TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
        HOME_LINE="Environment=HOME=${TARGET_HOME}"
        if ! grep -qF "$HOME_LINE" "$SERVICE_FILE"; then
            sudo sed -i "/^Environment=XAUTHORITY=/a ${HOME_LINE}" "$SERVICE_FILE"
            ok "Added Environment=HOME=$TARGET_HOME"
        else
            ok "HOME already set in service — no change"
        fi
    fi

    # ── 2d. Ensure num2words is installed in venv ──────────────────────────────
    if [ -f "$REPO_DIR/.venv/bin/pip" ]; then
        if ! "$REPO_DIR/.venv/bin/python3" -c "import num2words" 2>/dev/null; then
            info "Installing num2words in .venv…"
            "$REPO_DIR/.venv/bin/pip" install --prefer-binary num2words 2>/dev/null || true
            ok "num2words installed"
        else
            ok "num2words already installed in .venv"
        fi
    fi

    # ── 3. Write Pi4 .env settings ───────────────────────────────────────────
    info "Checking .env for Pi4 optimisation settings…"
    touch "$ENV_FILE"

    _env_set() {
        local key="$1" val="$2"
        if grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
            ok "${key} already set — skipping"
        else
            echo "${key}=${val}" >> "$ENV_FILE"
            ok "Added ${key}=${val}"
        fi
    }

    if ! grep -q "Pi 4 Performance" "$ENV_FILE" 2>/dev/null; then
        printf '\n# ── Raspberry Pi 4 Performance Optimisations ──\n' >> "$ENV_FILE"
    fi

    _env_set CTX_SIZE                   2048
    _env_set N_BATCH                    256
    _env_set SPECULATIVE_DECODING       prompt_lookup
    _env_set SPECULATIVE_NGRAM_SIZE     3
    _env_set SPECULATIVE_NUM_PRED_TOKENS 10
    _env_set USE_KOKORO_ONNX            true
    _env_set VLM_ENABLED                0
    _env_set FACE_RECOGNITION_INTERVAL  2.0
fi

# ── 4. Reload + restart ───────────────────────────────────────────────────────
if $DO_RESTART; then
    if [ -f "$SERVICE_FILE" ]; then
        info "Reloading systemd and restarting karma.service…"
        sudo systemctl daemon-reload
        sudo systemctl restart karma.service
        sleep 2
        STATUS=$(systemctl is-active karma.service 2>/dev/null || echo "unknown")
        if [ "$STATUS" = "active" ]; then
            ok "karma.service is running ✦"
        else
            warn "karma.service status: $STATUS"
            warn "Check logs:  journalctl -fu karma.service"
        fi
    else
        warn "karma.service not found — nothing to restart"
    fi
else
    info "Skipping restart (--no-restart). Apply manually:"
    echo "       sudo systemctl daemon-reload && sudo systemctl restart karma.service"
fi

# ── 5. Done ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}Patch complete.${NC}"
if $DO_RESTART && systemctl is-active --quiet karma.service 2>/dev/null; then
    echo ""
    echo "  Tail live logs:   journalctl -fu karma.service"
    echo "  Or watch file:    tail -f $REPO_DIR/karma.log"
fi
echo ""
