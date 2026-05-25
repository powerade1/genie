import subprocess
import time

import pyperclip

from config import CURSOR_APP_NAME, CURSOR_MATCH_MODE_ON_APPROVE
from supervisor.cursor_modes import focus_composer_input, match_composer_mode, normalize_mode

_KEY_DELAY_SEC = 0.35
_COMPOSER_OPEN_DELAY_SEC = 0.5


def _run_osascript(script: str) -> None:
    subprocess.run(
        ["osascript", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )


def _cursor_action(body: str) -> None:
    """Cursor가 이미 활성화된 상태에서 키 입력만 전송."""
    script = f"""
tell application "System Events"
    if not (exists process "{CURSOR_APP_NAME}") then error "Cursor process not found"
    tell process "{CURSOR_APP_NAME}"
        {body}
    end tell
end tell
"""
    _run_osascript(script)


def _cursor_keystroke(key: str, modifiers: list[str] | None = None) -> None:
    if modifiers:
        mod_str = ", ".join(f"{mod} down" for mod in modifiers)
        _cursor_action(f'keystroke "{key}" using {{{mod_str}}}')
    else:
        _cursor_action(f'keystroke "{key}"')


def _cursor_key_code(code: int, modifiers: list[str] | None = None) -> None:
    if modifiers:
        mod_str = ", ".join(f"{mod} down" for mod in modifiers)
        _cursor_action(f"key code {code} using {{{mod_str}}}")
    else:
        _cursor_action(f"key code {code}")


def _activate_cursor() -> None:
    _run_osascript(f'tell application "{CURSOR_APP_NAME}" to activate')
    time.sleep(_KEY_DELAY_SEC)


def _prepare_composer() -> None:
    """Cursor 활성화 → 활성 Composer 입력 포커스 (Cmd+I는 최후 수단)."""
    _activate_cursor()

    # 에디터 자동완성 등만 닫고, Composer/세션 상태는 유지
    _cursor_key_code(53)  # Escape
    time.sleep(0.15)

    if focus_composer_input():
        time.sleep(_KEY_DELAY_SEC)
        return

    # Composer 패널이 닫혀 있을 때만 Cmd+I (토글로 세션/패널이 바뀌는 부작용 방지)
    _cursor_keystroke("i", ["command"])
    time.sleep(_COMPOSER_OPEN_DELAY_SEC)

    if not focus_composer_input():
        raise RuntimeError(
            "Cursor Composer 입력창을 찾지 못했습니다. "
            "Composer가 열려 있는지, 손쉬운 사용 권한이 허용됐는지 확인하세요."
        )
    time.sleep(_KEY_DELAY_SEC)


def send_to_cursor(prompt: str, target_mode: str = "plan") -> None:
    """프롬프트를 붙여넣고 target_mode에 맞춘 뒤 Composer에 제출."""
    if not prompt.strip():
        raise ValueError("프롬프트가 비어 있습니다.")

    mode = normalize_mode(target_mode) or target_mode.strip().lower() or "plan"

    pyperclip.copy(prompt)
    _prepare_composer()

    # 모드 매칭 실패해도 붙여넣기는 진행 (Shift+Tab 과다 시도 방지)
    if CURSOR_MATCH_MODE_ON_APPROVE:
        match_composer_mode(mode)

    _cursor_keystroke("v", ["command"])
    time.sleep(_KEY_DELAY_SEC)
    _cursor_key_code(36)  # Return
