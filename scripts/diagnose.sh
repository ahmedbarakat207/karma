#!/usr/bin/env bash
# Run on the Pi to collect all startup debug info in one shot.
# Usage:  bash scripts/diagnose.sh 2>&1 | tee /tmp/karma_diag.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICE_FILE="/etc/systemd/system/karma.service"
SEP="────────────────────────────────────────"

h() { echo ""; echo "=== $1 ==="; echo "$SEP"; }

h "SERVICE STATUS"
systemctl status karma.service --no-pager -l 2>&1 || echo "(service not found)"

h "LAST 40 JOURNAL LINES"
journalctl -u karma.service -n 40 --no-pager 2>&1 || echo "(no journal)"

h "KARMA.LOG (last 60 lines)"
KLOG="$REPO_DIR/karma.log"
if [ -f "$KLOG" ]; then
    tail -n 60 "$KLOG"
else
    echo "karma.log not found at $KLOG"
    # Look for it anywhere nearby
    find "$REPO_DIR" -maxdepth 1 -name "*.log" 2>/dev/null
fi

h "SHELL STARTUP LOG (last 30 lines)"
SLOG="$REPO_DIR/start_robot.log"
[ -f "$SLOG" ] && tail -n 30 "$SLOG" || echo "(no start_robot.log — will exist after next start)"

h "SERVICE FILE"
[ -f "$SERVICE_FILE" ] && cat "$SERVICE_FILE" || echo "(service file not found)"

h "START_ROBOT.SH (first 20 lines)"
[ -f "$REPO_DIR/start_robot.sh" ] && head -n 20 "$REPO_DIR/start_robot.sh" || echo "(not found)"

h "PYTHON / ENV"
echo "DISPLAY: ${DISPLAY:-not set}"
echo "XAUTHORITY: ${XAUTHORITY:-not set}"
echo "HOME: ${HOME:-not set}"
PYTHON_BIN=""
[ -f "$REPO_DIR/.venv/bin/python3" ] && PYTHON_BIN="$REPO_DIR/.venv/bin/python3"
[ -z "$PYTHON_BIN" ] && PYTHON_BIN="$(command -v python3 2>/dev/null || echo 'NOT FOUND')"
echo "Python bin: $PYTHON_BIN"
"$PYTHON_BIN" --version 2>&1 || true
echo "nice test: $(nice -n -5 true 2>&1 && echo OK || echo EPERM)"
echo "taskset:   $(command -v taskset 2>/dev/null || echo 'not found')"

h "DOT ENV"
[ -f "$REPO_DIR/.env" ] && grep -v "API_KEY" "$REPO_DIR/.env" || echo "(.env not found)"

h "GIT STATUS"
cd "$REPO_DIR" && git log --oneline -5 && echo "" && git status --short

h "DISK / MEMORY"
df -h "$REPO_DIR" 2>/dev/null | tail -1
free -h 2>/dev/null | head -2

echo ""
echo "=== DONE — share everything above ==="
