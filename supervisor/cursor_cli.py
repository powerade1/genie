import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CURSOR_CLI_PATH,
    CURSOR_WORKSPACE,
)
from supervisor.cursor_modes import normalize_mode

_DEFAULT_CLI_PATH = "/Applications/Cursor.app/Contents/Resources/app/bin/cursor"

_MODE_LABELS: dict[str, str] = {
    "plan": "Plan",
    "ask": "Ask",
    "agent": "Agent",
    "debug": "Debug",
}


def resolve_cli_path() -> Path:
    configured = (CURSOR_CLI_PATH or "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return path

    found = shutil.which("cursor")
    if found:
        return Path(found)

    fallback = Path(_DEFAULT_CLI_PATH)
    if fallback.is_file():
        return fallback

    raise FileNotFoundError(
        "Cursor CLI를 찾을 수 없습니다. "
        "Cursor → Command Palette → 'Install cursor command'로 설치하거나 "
        f"config.py의 CURSOR_CLI_PATH를 설정하세요. (기본 경로: {_DEFAULT_CLI_PATH})"
    )


def resolve_workspace() -> Path:
    configured = (CURSOR_WORKSPACE or "").strip()
    if configured:
        workspace = Path(configured).expanduser().resolve()
    else:
        workspace = PROJECT_ROOT.resolve()

    if not workspace.is_dir():
        raise FileNotFoundError(f"워크스페이스 경로가 존재하지 않습니다: {workspace}")

    return workspace


def build_agent_command(
    prompt: str,
    target_mode: str,
    *,
    continue_session: bool = False,
) -> list[str]:
    if not prompt.strip():
        raise ValueError("프롬프트가 비어 있습니다.")

    cli_path = resolve_cli_path()
    workspace = resolve_workspace()
    mode = normalize_mode(target_mode) or target_mode.strip().lower() or "plan"

    cmd = [str(cli_path), "agent", "--workspace", str(workspace), "--trust"]
    cmd.extend(["--print", "--output-format", "stream-json", "--stream-partial-output"])

    if continue_session:
        # 직전 세션 이어가기 — 모드는 세션에 종속되므로 모드 플래그 생략
        cmd.append("--continue")
    else:
        if mode == "plan":
            cmd.append("--plan")
        elif mode == "ask":
            cmd.extend(["--mode", "ask"])
        # agent / debug: CLI에 별도 플래그 없음 → 기본 에이전트 모드

    cmd.append(prompt)
    return cmd


def mode_label(target_mode: str) -> str:
    mode = normalize_mode(target_mode) or target_mode.strip().lower() or "plan"
    return _MODE_LABELS.get(mode, mode)
