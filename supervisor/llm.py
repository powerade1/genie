import json
import os
from typing import TypedDict

from google import genai
from google.genai import types

from config import (
    SUGGEST_PROMPT_INSTRUCTION,
    SUPERVISOR_MODEL,
)
from supervisor.context import build_system_instruction

_client: genai.Client | None = None
_chat = None
_message_log: list[tuple[str, str]] = []

VALID_TARGET_MODES = ("agent", "plan", "debug", "ask")


class ChatResponse(TypedDict):
    reply: str
    thinking: str
    ready_for_prompt: bool


class SuggestResponse(TypedDict):
    cursor_prompt: str
    target_mode: str


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
    global _chat, _message_log
    _chat = _create_chat()
    _message_log = []


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


def _format_history() -> str:
    if not _message_log:
        return "(대화 없음)"
    lines = []
    for role, content in _message_log:
        label = "사용자" if role == "user" else "Supervisor"
        lines.append(f"{label}: {content}")
    return "\n\n".join(lines)


def chat_send(message: str) -> ChatResponse:
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


def suggest_prompt() -> SuggestResponse:
    if not _message_log:
        raise ValueError("대화가 없습니다. 먼저 Supervisor와 대화하세요.")

    history = _format_history()
    prompt = (
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
