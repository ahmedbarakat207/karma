#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHELL_LOG="$SCRIPT_DIR/start_robot.log"

# Shell-level startup log — separate from karma.log (Python output).
# If karma.log is empty but start_robot.log has entries, Python is crashing on import.
# If start_robot.log is missing entirely, the shell script itself never ran.
_slog() { echo "[$(date '+%H:%M:%S')] $*" >> "$SHELL_LOG"; }
_slog "──────────────────────────────────"
_slog "start_robot.sh launched (PID $$)"

# ── Display: make sure X is actually ready before we connect ──────────────────
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
_slog "DISPLAY=$DISPLAY  XAUTHORITY=$XAUTHORITY  HOME=${HOME:-<unset>}"

# Wait up to 15 s for the X server to accept connections
_x_wait=0
until xset q &>/dev/null || [ $_x_wait -ge 15 ]; do
    sleep 1; _x_wait=$((_x_wait + 1))
done
_slog "X ready after ${_x_wait}s (xset q=$(xset q &>/dev/null && echo OK || echo FAIL))"

xset s off    2>/dev/null || true
xset -dpms    2>/dev/null || true
xset s noblank 2>/dev/null || true

unclutter -idle 0.1 -root &

openbox &

# ── Python binary ────────────────────────────────────────────────────────────
PYTHON_BIN=""
if [ -f "$SCRIPT_DIR/.venv/bin/python3" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
else
    PYTHON_BIN="$(command -v python3 2>/dev/null || true)"
fi

if [ -z "$PYTHON_BIN" ]; then
    _slog "ERROR: python3 not found — cannot start"
    exit 1
fi
_slog "Python: $PYTHON_BIN  ($("$PYTHON_BIN" --version 2>&1))"

while true; do
    _slog "launching main.py…"
    "$PYTHON_BIN" "$SCRIPT_DIR/main.py" >> "$SCRIPT_DIR/karma.log" 2>&1
    _slog "main.py exited (code $?), restarting in 3 s…"
    echo "[start_robot] process exited, restarting in 3s…" >> "$SCRIPT_DIR/karma.log"
    sleep 3
done
