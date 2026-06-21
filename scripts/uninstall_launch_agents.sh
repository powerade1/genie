#!/bin/bash
# genie LaunchAgent 제거 스크립트

set -euo pipefail

AGENT_DIR="$HOME/Library/LaunchAgents"
UID_NUM="$(id -u)"

GENIE_LABEL="com.woojin.genie"
WAKE_LABEL="com.woojin.genie.wake"
LOGVIEW_LABEL="com.woojin.genie.logview"

bootout_if_loaded() {
    local label="$1"
    if launchctl print "gui/$UID_NUM/$label" >/dev/null 2>&1; then
        echo "[uninstall] $label bootout"
        launchctl bootout "gui/$UID_NUM/$label" || true
    fi
}

bootout_if_loaded "$GENIE_LABEL"
bootout_if_loaded "$WAKE_LABEL"
bootout_if_loaded "$LOGVIEW_LABEL"

rm -f \
    "$AGENT_DIR/$GENIE_LABEL.plist" \
    "$AGENT_DIR/$WAKE_LABEL.plist" \
    "$AGENT_DIR/$LOGVIEW_LABEL.plist"
echo "[uninstall] 완료"
