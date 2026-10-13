#!/bin/bash
# genie LaunchAgent 설치 스크립트
#
# 동작:
#   1) ~/Library/LaunchAgents/ 에 plist 를 복사
#   2) 이미 등록되어 있으면 bootout 후 다시 bootstrap
#   3) wake watcher / genie 본체 / (기본) logview 를 즉시 kickstart
#
# 옵션:
#   --no-logview     로그 뷰어 Terminal 창 자동 띄우기 비활성 (프로덕션 모드)
#   --logview        명시적으로 활성 (기본값과 동일)
#
# 제거: scripts/uninstall_launch_agents.sh 실행

set -euo pipefail

WITH_LOGVIEW=1
for arg in "$@"; do
    case "$arg" in
        --no-logview) WITH_LOGVIEW=0 ;;
        --logview)    WITH_LOGVIEW=1 ;;
        *) echo "[install] 알 수 없는 옵션: $arg" >&2; exit 2 ;;
    esac
done

PROJECT_DIR="/Users/woojin/Desktop/DEV/2_active/genie"
AGENT_DIR="$HOME/Library/LaunchAgents"
UID_NUM="$(id -u)"

GENIE_LABEL="com.woojin.genie"
WAKE_LABEL="com.woojin.genie.wake"
LOGVIEW_LABEL="com.woojin.genie.logview"

GENIE_SRC="$PROJECT_DIR/scripts/$GENIE_LABEL.plist"
WAKE_SRC="$PROJECT_DIR/scripts/$WAKE_LABEL.plist"
LOGVIEW_SRC="$PROJECT_DIR/scripts/$LOGVIEW_LABEL.plist"
GENIE_DST="$AGENT_DIR/$GENIE_LABEL.plist"
WAKE_DST="$AGENT_DIR/$WAKE_LABEL.plist"
LOGVIEW_DST="$AGENT_DIR/$LOGVIEW_LABEL.plist"

chmod +x "$PROJECT_DIR/scripts/show_logs.sh" 2>/dev/null || true

echo "[install] LaunchAgent 디렉토리 준비: $AGENT_DIR"
mkdir -p "$AGENT_DIR"
mkdir -p "$PROJECT_DIR/logs"

# --- .env 자동 생성 -----------------------------------------------------------
# launchd 는 ~/.zshrc 같은 shell rc 파일을 읽지 않으므로, 프로젝트의 .env 에
# 비밀을 모아두고 main.py 가 직접 로드하도록 한다.
ENV_FILE="$PROJECT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "[install] .env 가 없습니다. ~/.zshrc 등에서 키를 추출해 자동 생성 시도"
    : > "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    for var in GEMINI_API_KEY TAVILY_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY; do
        # zsh 를 사용해 사용자의 rc 파일을 source 한 뒤 변수 값만 출력
        val=$(/bin/zsh -c '
            for f in "$HOME/.zshenv" "$HOME/.zprofile" "$HOME/.zshrc"; do
                [ -f "$f" ] && . "$f" >/dev/null 2>&1
            done
            printf "%s" "${'"$var"'}"
        ' 2>/dev/null || true)
        if [ -n "$val" ]; then
            printf "%s=%s\n" "$var" "$val" >> "$ENV_FILE"
            echo "  + $var (from shell rc)"
        fi
    done
    if [ ! -s "$ENV_FILE" ]; then
        echo "  ! 추출된 키가 없습니다. $ENV_FILE 에 직접 추가하세요. 예:"
        echo "      echo 'GEMINI_API_KEY=your-key' >> $ENV_FILE"
    fi
fi
# ---------------------------------------------------------------------------

bootout_if_loaded() {
    local label="$1"
    if launchctl print "gui/$UID_NUM/$label" >/dev/null 2>&1; then
        echo "[install] 기존 $label bootout"
        launchctl bootout "gui/$UID_NUM/$label" || true
    fi
}

bootout_if_loaded "$GENIE_LABEL"
bootout_if_loaded "$WAKE_LABEL"
bootout_if_loaded "$LOGVIEW_LABEL"

echo "[install] plist 복사"
cp "$GENIE_SRC" "$GENIE_DST"
cp "$WAKE_SRC" "$WAKE_DST"
PYTHON="/Users/woojin/anaconda3/envs/genie/bin/python"
LAUNCH_FLAGS=$("$PYTHON" "$PROJECT_DIR/scripts/launch_config.py" flags 2>/dev/null || echo "boot=1 wake=1")
BOOT_AUTO=${LAUNCH_FLAGS#*boot=}
BOOT_AUTO=${BOOT_AUTO%% *}
WAKE_AUTO=${LAUNCH_FLAGS#*wake=}
echo "[install] config: LAUNCH_AGENT_AUTO_START_ON_BOOT=$([ "$BOOT_AUTO" = 1 ] && echo true || echo false)"
echo "[install] config: LAUNCH_AGENT_AUTO_START_ON_WAKE=$([ "$WAKE_AUTO" = 1 ] && echo true || echo false)"
"$PYTHON" "$PROJECT_DIR/scripts/launch_config.py" apply-plist "$GENIE_DST"
if [ "$WITH_LOGVIEW" = "1" ]; then
    cp "$LOGVIEW_SRC" "$LOGVIEW_DST"
    "$PYTHON" "$PROJECT_DIR/scripts/launch_config.py" apply-plist "$LOGVIEW_DST" "$BOOT_AUTO"
else
    rm -f "$LOGVIEW_DST"
fi

echo "[install] bootstrap"
launchctl bootstrap "gui/$UID_NUM" "$GENIE_DST"
launchctl bootstrap "gui/$UID_NUM" "$WAKE_DST"
if [ "$WITH_LOGVIEW" = "1" ]; then
    launchctl bootstrap "gui/$UID_NUM" "$LOGVIEW_DST"
fi

echo "[install] enable (영구 등록)"
launchctl enable "gui/$UID_NUM/$GENIE_LABEL" || true
launchctl enable "gui/$UID_NUM/$WAKE_LABEL" || true
if [ "$WITH_LOGVIEW" = "1" ]; then
    launchctl enable "gui/$UID_NUM/$LOGVIEW_LABEL" || true
fi

echo "[install] kickstart"
launchctl kickstart -k "gui/$UID_NUM/$WAKE_LABEL" || true
if [ "$BOOT_AUTO" = "1" ]; then
    launchctl kickstart -k "gui/$UID_NUM/$GENIE_LABEL" || true
else
    echo "[install] boot 자동 시작 꺼짐 → genie kickstart 생략"
fi
if [ "$WITH_LOGVIEW" = "1" ] && { [ "$BOOT_AUTO" = "1" ] || [ "$WAKE_AUTO" = "1" ]; }; then
    launchctl kickstart -k "gui/$UID_NUM/$LOGVIEW_LABEL" || true
elif [ "$WITH_LOGVIEW" = "1" ]; then
    echo "[install] boot/wake 자동 시작 모두 꺼짐 → logview kickstart 생략"
fi

sleep 2

echo
echo "[install] 상태:"
launchctl print "gui/$UID_NUM/$GENIE_LABEL" 2>/dev/null | grep -E "state|last exit code|pid" || true
echo "---"
launchctl print "gui/$UID_NUM/$WAKE_LABEL" 2>/dev/null | grep -E "state|last exit code|pid" || true
if [ "$WITH_LOGVIEW" = "1" ]; then
    echo "---"
    launchctl print "gui/$UID_NUM/$LOGVIEW_LABEL" 2>/dev/null | grep -E "state|last exit code|pid" || true
fi

# 권한(TCC) 거부 자동 진단 — last exit code 가 126/127/1 이거나 stderr 에 "Operation not permitted" 있으면 안내
GENIE_EXIT=$(launchctl print "gui/$UID_NUM/$GENIE_LABEL" 2>/dev/null | awk -F'=' '/last exit code/ {gsub(/ /,"",$2); print $2; exit}')
if grep -q "Operation not permitted" "$PROJECT_DIR/logs/genie.err.log" 2>/dev/null; then
    NEEDS_TCC=1
elif [ "$GENIE_EXIT" = "126" ] || [ "$GENIE_EXIT" = "127" ]; then
    NEEDS_TCC=1
else
    NEEDS_TCC=0
fi

echo
echo "로그:"
echo "  tail -f $PROJECT_DIR/logs/genie.out.log"
echo "  tail -f $PROJECT_DIR/logs/genie.err.log"
echo "  tail -f $PROJECT_DIR/logs/wake.out.log"
if [ "$WITH_LOGVIEW" = "1" ]; then
    echo
    if [ "$BOOT_AUTO" = "1" ] || [ "$WAKE_AUTO" = "1" ]; then
        echo "라이브 로그 뷰어가 새 Terminal 창으로 떠 있어야 합니다."
        echo "  창을 닫았을 때 다시 띄우려면:"
        echo "    launchctl kickstart -k gui/$UID_NUM/$LOGVIEW_LABEL"
    else
        echo "boot/wake 자동 시작이 모두 꺼져 있어 logview 자동 실행을 건너뜄습니다."
    fi
    echo "  자동 띄우기 끄려면:  bash $0 --no-logview"
fi

if [ "$NEEDS_TCC" = "1" ]; then
    echo
    echo "============================================================"
    echo " 권한 부족: macOS TCC 가 launchd 의 ~/Desktop 접근을 막고 있습니다."
    echo " 시스템 설정 → 개인정보 보호 및 보안 → '전체 디스크 액세스' 패널을 열고"
    echo " 아래 바이너리를 + 로 추가한 뒤 토글을 켜주세요:"
    echo "   /Users/woojin/anaconda3/envs/genie/bin/python"
    echo
    echo " (Finder 에서 Cmd+Shift+G → 위 경로 입력 → 선택)"
    echo " 추가 후:  bash $PROJECT_DIR/scripts/install_launch_agents.sh  를 다시 실행"
    echo "============================================================"
    open "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles" 2>/dev/null || true
fi

echo
echo "주의: 시스템 설정 → 일반 → 로그인 항목 에 있던 기존 genie/start_voice.sh 자동 실행 항목은 제거하세요."
