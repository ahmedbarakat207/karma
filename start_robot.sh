#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Display: make sure X is actually ready before we try to connect ──────────
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

# Wait up to 15 s for the X server to accept connections (avoids "cannot open display" crashes)
_x_wait=0
until xset q &>/dev/null || [ $_x_wait -ge 15 ]; do
    sleep 1; _x_wait=$((_x_wait + 1))
done

xset s off    2>/dev/null || true
xset -dpms    2>/dev/null || true
xset s noblank 2>/dev/null || true

unclutter -idle 0.1 -root &

openbox &

# ── Determine runner prefix ──────────────────────────────────────────────────
# nice -n -5 requires CAP_SYS_NICE or root on some Pi OS images.
# We try it; if nice exits 1 (EPERM) we fall back to plain nice -n 0.
PYTHON_BIN=""
if [ -f "$SCRIPT_DIR/.venv/bin/python3" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
else
    PYTHON_BIN="$(command -v python3)"
fi

_nice_ok=0
nice -n -5 true 2>/dev/null && _nice_ok=1

while true; do
    if [ $_nice_ok -eq 1 ] && command -v taskset &>/dev/null; then
        nice -n -5 taskset -c 0-3 "$PYTHON_BIN" "$SCRIPT_DIR/main.py" >> "$SCRIPT_DIR/karma.log" 2>&1
    elif [ $_nice_ok -eq 1 ]; then
        nice -n -5 "$PYTHON_BIN" "$SCRIPT_DIR/main.py" >> "$SCRIPT_DIR/karma.log" 2>&1
    else
        "$PYTHON_BIN" "$SCRIPT_DIR/main.py" >> "$SCRIPT_DIR/karma.log" 2>&1
    fi
    echo "[start_robot] process exited, restarting in 3s…" >> "$SCRIPT_DIR/karma.log"
    sleep 3
done
