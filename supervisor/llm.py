import json
import os
from typing import TypedDict

from google import genai
from google.genai import types

from config import (
    CURSOR_FOLLOWUP_INSTRUCTION,
    SUGGEST_PROMPT_INSTRUCTION,
    SUPERVISOR_MODEL,
)
from supervisor.context import build_system_instruction

_client: genai.Client | None = None
_chat = None
_message_log: list[tuple[str, str]] = []
_executed_log_end: int = 0

VALID_TARGET_MODES = ("agent", "plan", "debug", "ask")


class ChatResponse(TypedDict):
    reply: str
    thinking: str
    ready_for_prompt: bool


class SuggestResponse(TypedDict):
    cursor_prompt: str
    target_mode: str


class FollowupResponse(TypedDict):
    action: str  # "ask_user" | "auto_continue" | "finish"
    question: str
    auto_answer: str
    reasoning: str


VALID_FOLLOWUP_ACTIONS = ("ask_user", "auto_continue", "finish")


CHAT_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "thinking": types.Schema(
            type=types.Type.STRING,
            description="질문 의도 분석 및 내부 논리 도출 (한국어)",
        ),
        "reply": types.Schema(
            type=types.Type.STRING,
            description="사용자와의 대화 응답 (한국어, Markdown)",
        ),
        "ready_for_prompt": types.Schema(
            type=types.Type.BOOLEAN,
            description="유저 최종 동의·실행 의도 시 true",
        ),
    },
    required=["thinking", "reply", "ready_for_prompt"],
)

SUGGEST_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "cursor_prompt": types.Schema(
            type=types.Type.STRING,
            description="Cursor Composer용 프롬프트 (한국어 Markdown)",
        ),
        "target_mode": types.Schema(
            type=types.Type.STRING,
            enum=list(VALID_TARGET_MODES),
            description="Cursor Composer 실행 모드",
        ),
    },
    required=["cursor_prompt", "target_mode"],
)

FOLLOWUP_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "action": types.Schema(
            type=types.Type.STRING,
            enum=list(VALID_FOLLOWUP_ACTIONS),
            description="후속 액션 (ask_user | auto_continue | finish)",
        ),
        "question": types.Schema(
            type=types.Type.STRING,
            description="ask_user일 때 사용자에게 보여줄 한국어 질문",
        ),
        "auto_answer": types.Schema(
            type=types.Type.STRING,
            description="auto_continue일 때 Cursor에 보낼 한국어 답변",
        ),
        "reasoning": types.Schema(
            type=types.Type.STRING,
            description="판단 근거 (한국어 간결)",
        ),
    },
    required=["action", "reasoning"],
)


def ensure_api_key() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        raise EnvironmentError(
            "GEMINI_API_KEY 환경변수가 설정되지 않았습니다.\n"
            "export GEMINI_API_KEY='your-api-key'"
        )


def get_client() -> genai.Client:
    global _client
    if _client is None:
        ensure_api_key()
        _client = genai.Client()
    return _client


def _create_chat():
    return get_client().chats.create(
        model=SUPERVISOR_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=build_system_instruction(),
            response_mime_type="application/json",
            response_schema=CHAT_SCHEMA,
        ),
    )


def reset_chat() -> None:
    global _chat, _message_log, _executed_log_end
    _chat = _create_chat()
    _message_log = []
    _executed_log_end = 0


def _get_chat():
    global _chat
    if _chat is None:
        _chat = _create_chat()
    return _chat


def _parse_chat_response(text: str) -> ChatResponse:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Supervisor JSON 파싱 실패: {error}") from error

    return {
        "reply": data.get("reply", ""),
        "thinking": (data.get("thinking") or "").strip(),
        "ready_for_prompt": bool(data.get("ready_for_prompt", False)),
    }


def _normalize_target_mode(raw: str | None) -> str:
    mode = (raw or "plan").strip().lower()
    if mode in VALID_TARGET_MODES:
        return mode
    return "plan"


def _parse_suggest_response(text: str) -> SuggestResponse:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"프롬프트 JSON 파싱 실패: {error}") from error

    cursor_prompt = (data.get("cursor_prompt") or "").strip()
    if not cursor_prompt:
        raise ValueError("생성된 프롬프트가 비어 있습니다.")

    return {
        "cursor_prompt": cursor_prompt,
        "target_mode": _normalize_target_mode(data.get("target_mode")),
    }


def _format_history(messages: list[tuple[str, str]] | None = None) -> str:
    log = _message_log if messages is None else messages
    if not log:
        return "(대화 없음)"
    lines = []
    for role, content in log:
        label = "사용자" if role == "user" else "Supervisor"
        lines.append(f"{label}: {content}")
    return "\n\n".join(lines)


def _active_message_log() -> list[tuple[str, str]]:
    return _message_log[_executed_log_end:]


def chat_send(message: str, *, cursor_execution_completed: bool = False) -> ChatResponse:
    global _executed_log_end

    if cursor_execution_completed:
        _executed_log_end = len(_message_log)

    chat = _get_chat()
    _message_log.append(("user", message))
    response = chat.send_message(message)
    text = response.text
    if not text:
        raise ValueError("Supervisor 응답이 비어 있습니다.")
    result = _parse_chat_response(text)
    _message_log.append(("model", result["reply"]))
    return result


def seed_chat(context: str) -> None:
    """세션 이어하기: 요약 context를 대화 로그에만 기록 (LLM 호출 없음)."""
    _message_log.append(("user", f"[이전 대화 요약]\n{context}"))


def _parse_followup_response(text: str) -> FollowupResponse:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Followup JSON 파싱 실패: {error}") from error

    action = (data.get("action") or "").strip().lower()
    if action not in VALID_FOLLOWUP_ACTIONS:
        action = "finish"

    return {
        "action": action,
        "question": (data.get("question") or "").strip(),
        "auto_answer": (data.get("auto_answer") or "").strip(),
        "reasoning": (data.get("reasoning") or "").strip(),
    }


def analyze_cursor_followup(
    *,
    original_prompt: str,
    target_mode: str,
    cursor_output: str,
    previous_rounds: list[dict] | None = None,
) -> FollowupResponse:
    """Cursor Agent 마지막 응답을 분석해 후속 액션을 결정한다."""
    cursor_output = (cursor_output or "").strip()
    if not cursor_output:
        return {
            "action": "finish",
            "question": "",
            "auto_answer": "",
            "reasoning": "Cursor 응답이 비어 있어 종료.",
        }

    rounds_text = ""
    if previous_rounds:
        rendered = []
        for idx, round_info in enumerate(previous_rounds, start=1):
            kind = round_info.get("kind", "?")
            text = (round_info.get("text") or "").strip()
            if not text:
                continue
            rendered.append(f"[라운드 {idx} · {kind}]\n{text}")
        if rendered:
            rounds_text = "## 이전 라운드\n" + "\n\n".join(rendered) + "\n\n"

    prompt = (
        f"## 최초 프롬프트 ({target_mode} 모드)\n{original_prompt}\n\n"
        f"{rounds_text}"
        f"## Cursor 마지막 응답\n{cursor_output}\n\n"
        f"## 지시\n{CURSOR_FOLLOWUP_INSTRUCTION}"
    )

    response = get_client().models.generate_content(
        model=SUPERVISOR_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=build_system_instruction(),
            response_mime_type="application/json",
            response_schema=FOLLOWUP_SCHEMA,
        ),
    )

    text = response.text
    if not text:
        raise ValueError("Followup 분석 응답이 비어 있습니다.")

    return _parse_followup_response(text)


def suggest_prompt() -> SuggestResponse:
    active_log = _active_message_log()
    if not active_log:
        raise ValueError("대화가 없습니다. 먼저 Supervisor와 대화하세요.")

    history = _format_history(active_log)
    executed_note = ""
    if _executed_log_end > 0:
        executed_note = (
            "## 참고\n"
            "이전 대화에서 합의된 작업은 Cursor에서 이미 실행되었습니다. "
            "아래 기록만 반영하고, 완료된 작업을 프롬프트에 중복 포함하지 마세요.\n\n"
        )

    prompt = (
        f"{executed_note}"
        f"## 대화 기록\n{history}\n\n"
        f"## 지시\n{SUGGEST_PROMPT_INSTRUCTION}"
    )

    response = get_client().models.generate_content(
        model=SUPERVISOR_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=build_system_instruction(),
            response_mime_type="application/json",
            response_schema=SUGGEST_SCHEMA,
        ),
    )

    text = response.text
    if not text:
        raise ValueError("프롬프트 생성 응답이 비어 있습니다.")

    return _parse_suggest_response(text)
