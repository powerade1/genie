#!/bin/bash
# config.py 의 LaunchAgent 자동 실행 설정을 이미 설치된 plist 에 반영한다.
#
# 사용:
#   config.py 에서 LAUNCH_AGENT_AUTO_START_ON_BOOT / _ON_WAKE 수정
#   bash scripts/sync_launch_config.sh
#
# - ON_BOOT: genie plist RunAtLoad 갱신 (+ bootstrap 재등록)
# - ON_WAKE: wake watcher 재시작 (새 설정 로드)

set -euo pipefail

PROJECT_DIR="/Users/woojin/Desktop/DEV/2_active/genie"
AGENT_DIR="$HOME/Library/LaunchAgents"
UID_NUM="$(id -u)"
PYTHON="/Users/woojin/anaconda3/envs/genie/bin/python"

GENIE_LABEL="com.woojin.genie"
WAKE_LABEL="com.woojin.genie.wake"
LOGVIEW_LABEL="com.woojin.genie.logview"
GENIE_DST="$AGENT_DIR/$GENIE_LABEL.plist"
WAKE_DST="$AGENT_DIR/$WAKE_LABEL.plist"
LOGVIEW_DST="$AGENT_DIR/$LOGVIEW_LABEL.plist"

if [ ! -f "$GENIE_DST" ]; then
    echo "[sync] LaunchAgent 미설치. 먼저: bash scripts/install_launch_agents.sh"
    exit 1
fi

LAUNCH_FLAGS=$("$PYTHON" "$PROJECT_DIR/scripts/launch_config.py" flags)
BOOT_AUTO=${LAUNCH_FLAGS#*boot=}
BOOT_AUTO=${BOOT_AUTO%% *}
WAKE_AUTO=${LAUNCH_FLAGS#*wake=}

echo "[sync] LAUNCH_AGENT_AUTO_START_ON_BOOT=$([ "$BOOT_AUTO" = 1 ] && echo true || echo false)"
echo "[sync] LAUNCH_AGENT_AUTO_START_ON_WAKE=$([ "$WAKE_AUTO" = 1 ] && echo true || echo false)"

"$PYTHON" "$PROJECT_DIR/scripts/launch_config.py" apply-plist "$GENIE_DST"

if [ -f "$LOGVIEW_DST" ]; then
    "$PYTHON" "$PROJECT_DIR/scripts/launch_config.py" apply-plist "$LOGVIEW_DST" "$BOOT_AUTO"
fi

if launchctl print "gui/$UID_NUM/$GENIE_LABEL" >/dev/null 2>&1; then
    launchctl bootout "gui/$UID_NUM/$GENIE_LABEL" || true
fi
launchctl bootstrap "gui/$UID_NUM" "$GENIE_DST"
launchctl enable "gui/$UID_NUM/$GENIE_LABEL" || true

if [ "$BOOT_AUTO" = "1" ]; then
    launchctl kickstart -k "gui/$UID_NUM/$GENIE_LABEL" || true
else
    echo "[sync] boot 자동 시작 꺼짐 (다음 로그인부터 genie 미시작)"
fi

if launchctl print "gui/$UID_NUM/$WAKE_LABEL" >/dev/null 2>&1; then
    launchctl kickstart -k "gui/$UID_NUM/$WAKE_LABEL" || true
    echo "[sync] wake watcher 재시작 (wake 설정 반영)"
fi

if [ -f "$LOGVIEW_DST" ]; then
    if [ "$BOOT_AUTO" = "1" ] || [ "$WAKE_AUTO" = "1" ]; then
        if launchctl print "gui/$UID_NUM/$LOGVIEW_LABEL" >/dev/null 2>&1; then
            launchctl kickstart -k "gui/$UID_NUM/$LOGVIEW_LABEL" || true
        else
            launchctl bootstrap "gui/$UID_NUM" "$LOGVIEW_DST"
            launchctl enable "gui/$UID_NUM/$LOGVIEW_LABEL" || true
            launchctl kickstart -k "gui/$UID_NUM/$LOGVIEW_LABEL" || true
        fi
        echo "[sync] logview 설정 반영"
    elif launchctl print "gui/$UID_NUM/$LOGVIEW_LABEL" >/dev/null 2>&1; then
        launchctl bootout "gui/$UID_NUM/$LOGVIEW_LABEL" || true
        echo "[sync] boot/wake 자동 시작 모두 꺼짐 → logview 중지"
    fi
fi

echo "[sync] 완료"
