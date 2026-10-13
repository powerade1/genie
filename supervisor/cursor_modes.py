import re
import subprocess
import time

from config import (
    CURSOR_APP_NAME,
    CURSOR_MODE_CYCLE,
    CURSOR_MODE_MATCH_MAX_STEPS,
    CURSOR_MODE_READ_DELAY_SEC,
)

_MODE_ALIASES: dict[str, list[str]] = {
    "agent": ["agent"],
    "plan": ["plan"],
    "debug": ["debug"],
    "multitask": ["multitask", "multi-task", "multi task"],
    "ask": ["ask"],
}

_READ_UI_SCRIPT = """
tell application "System Events"
    if not (exists process "{app_name}") then return ""
    tell process "{app_name}"
        if (count of windows) is 0 then return ""
        set collected to ""
        repeat with w in windows
            repeat with el in entire contents of w
                repeat with attr in {{value, name, description, title}}
                    try
                        set t to contents of attr of el
                        if t is not missing value then
                            set t to t as text
                            if t is not "" then set collected to collected & t & linefeed
                        end if
                    end try
                end repeat
            end repeat
        end repeat
        return collected
    end tell
end tell
"""

# 활성(현재) Composer 입력창 = 오른쪽 패널에서 Y가 가장 큰 텍스트 영역
_FOCUS_COMPOSER_SCRIPT = """
tell application "System Events"
    if not (exists process "{app_name}") then return "no_process"
    tell process "{app_name}"
        if (count of windows) is 0 then return "no_window"
        set frontWin to front window
        set winPos to position of frontWin
        set winSize to size of frontWin
        set winLeft to item 1 of winPos
        set winTop to item 2 of winPos
        set winWidth to item 1 of winSize
        set winHeight to item 2 of winSize
        set midX to winLeft + (winWidth * 0.45)

        try
            set focusedEl to value of attribute "AXFocusedUIElement" of frontWin
            if focusedEl is not missing value then
                try
                    set fRole to role of focusedEl
                    if fRole is "AXTextArea" or fRole is "AXTextField" then
                        set fX to item 1 of (position of focusedEl)
                        if fX >= midX then return "already_focused"
                    end if
                end try
            end if
        end try

        set activeEl to missing value
        set activeY to -1

        repeat with el in entire contents of frontWin
            try
                set elRole to role of el
                if elRole is "AXTextArea" or elRole is "AXTextField" then
                    set elPos to position of el
                    set elX to item 1 of elPos
                    set elY to item 2 of elPos
                    if elX >= midX and elY > activeY then
                        set activeY to elY
                        set activeEl to el
                    end if
                end if
            end try
        end repeat

        if activeEl is not missing value then
            try
                set value of attribute "AXFocused" of activeEl to true
            end try
            click activeEl
            return "clicked_active"
        end if

        set clickX to winLeft + (winWidth * 0.75)
        set clickY to winTop + (winHeight * 0.92)
        click at {{clickX, clickY}}
        return "clicked_coords"
    end tell
end tell
"""


def _run_osascript(script: str) -> str:
    result = subprocess.run(
        ["osascript", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return (result.stdout or "").strip()


def normalize_mode(raw: str) -> str | None:
    text = raw.strip().lower()
    if not text:
        return None
    for mode, aliases in _MODE_ALIASES.items():
        if text in aliases:
            return mode
    for mode, aliases in _MODE_ALIASES.items():
        for alias in aliases:
            if re.fullmatch(rf".*\b{re.escape(alias)}\b.*", text):
                return mode
    return None


def _read_cursor_ui_text() -> str:
    script = _READ_UI_SCRIPT.format(app_name=CURSOR_APP_NAME)
    try:
        return _run_osascript(script)
    except subprocess.CalledProcessError:
        return ""


def get_composer_mode() -> str | None:
    """Accessibility로 Cursor Composer 활성 모드 텍스트를 읽는다."""
    ui_text = _read_cursor_ui_text()
    if not ui_text:
        return None

    found: list[str] = []
    for line in ui_text.splitlines():
        mode = normalize_mode(line)
        if mode and mode in CURSOR_MODE_CYCLE:
            found.append(mode)

    if not found:
        return None

    return found[-1]


def focus_composer_input() -> bool:
    """현재(활성) Composer 입력창으로 포커스를 옮긴다. 세션 전환 없이 클릭."""
    script = _FOCUS_COMPOSER_SCRIPT.format(app_name=CURSOR_APP_NAME)
    try:
        result = _run_osascript(script)
    except subprocess.CalledProcessError:
        return False
    return result in {"already_focused", "clicked_active", "clicked_coords"}


def shift_tab_once() -> None:
    subprocess.run(
        [
            "osascript",
            "-e",
            f'tell application "System Events" to tell process "{CURSOR_APP_NAME}" '
            "to key code 48 using {shift down}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def match_composer_mode(target: str, max_steps: int | None = None) -> bool:
    """현재 Composer 모드를 target과 일치할 때까지 Shift+Tab으로 순환 (최대 1바퀴)."""
    target_norm = normalize_mode(target) or target.strip().lower()
    if target_norm not in CURSOR_MODE_CYCLE:
        target_norm = "plan"

    steps = max_steps if max_steps is not None else CURSOR_MODE_MATCH_MAX_STEPS
    steps = min(steps, len(CURSOR_MODE_CYCLE))

    seen: list[str] = []

    for attempt in range(steps + 1):
        current = get_composer_mode()
        if current == target_norm:
            return True
        if current and current in seen:
            break
        if current:
            seen.append(current)
        if attempt >= steps:
            break
        shift_tab_once()
        time.sleep(CURSOR_MODE_READ_DELAY_SEC)

    return get_composer_mode() == target_norm

