#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

xset s off 2>/dev/null || true
xset -dpms 2>/dev/null || true
xset s noblank 2>/dev/null || true

unclutter -idle 0.1 -root &

openbox &

while true; do
    # Pi 4 optimisation: slight priority boost + explicit CPU affinity so the
    # OS never migrates the LLM thread mid-inference. Falls back if taskset absent.
    RUNNER=""
    command -v taskset &>/dev/null && RUNNER="taskset -c 0-3 "
    RUNNER="nice -n -5 ${RUNNER}"

    if [ -f "$SCRIPT_DIR/.venv/bin/python3" ]; then
        ${RUNNER}"$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/main.py" >> "$SCRIPT_DIR/karma.log" 2>&1
    else
        ${RUNNER}python3 "$SCRIPT_DIR/main.py" >> "$SCRIPT_DIR/karma.log" 2>&1
    fi
    sleep 3
done
