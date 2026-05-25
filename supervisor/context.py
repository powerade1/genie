from pathlib import Path

from config import SUPERVISOR_CONTEXT_PATH, SUPERVISOR_SYSTEM_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _context_file_path() -> Path:
    path = Path(SUPERVISOR_CONTEXT_PATH)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_supervisor_context() -> str | None:
    context_path = _context_file_path()
    try:
        content = context_path.read_text(encoding="utf-8").strip()
        if not content:
            print(
                f"[Context Sync] {context_path.name}가 비어 있습니다. "
                "기본 컨텍스트로 진행합니다."
            )
            return None
        print(f"[Context Sync] {context_path.name} 로드 성공 (총 {len(content)}자)")
        return content
    except FileNotFoundError:
        print(
            f"[Context Sync] {context_path.name}를 찾을 수 없습니다. "
            "기본 컨텍스트로 진행합니다."
        )
        return None
    except OSError as error:
        print(
            f"[Context Sync] {context_path.name} 읽기 실패: {error}. "
            "기본 컨텍스트로 진행합니다."
        )
        return None


def build_system_instruction() -> str:
    base = SUPERVISOR_SYSTEM_PROMPT
    context = load_supervisor_context()
    if not context:
        return base
    return (
        f"{base}\n\n"
        "[Project Context — 기획 맥락 및 업데이트 히스토리]\n"
        "아래 문서는 프로젝트의 단일 진실 공급원(SSOT)이다. 답변·프롬프트 제안 시 이 맥락을 우선 참고하라.\n\n"
        f"{context}"
    )
