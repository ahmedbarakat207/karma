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
