"""Genie 연동용 Supervisor 세션 부트스트랩 저장."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from supervisor import llm

BOOTSTRAP_TTL_SECONDS = 3600


@dataclass
class BootstrapSession:
    session_id: str
    title: str
    messages: list[dict[str, Any]]
    ready_for_prompt: bool
    created_at: float = field(default_factory=time.time)


_bootstraps: dict[str, BootstrapSession] = {}


def _cleanup_expired() -> None:
    now = time.time()
    expired = [
        sid
        for sid, bootstrap in _bootstraps.items()
        if now - bootstrap.created_at > BOOTSTRAP_TTL_SECONDS
    ]
    for sid in expired:
        del _bootstraps[sid]


def _session_title(message: str) -> str:
    text = message.strip()
    if not text:
        return "새 대화"
    return text[:30] + ("…" if len(text) > 30 else "")


def create_session(message: str) -> BootstrapSession:
    """새 서버 채팅을 시작하고 첫 메시지를 보낸 뒤 부트스트랩을 저장한다."""
    _cleanup_expired()

    session_id = str(uuid.uuid4())
    llm.reset_chat()
    result = llm.chat_send(message)

    bootstrap = BootstrapSession(
        session_id=session_id,
        title=_session_title(message),
        messages=[
            {"role": "user", "content": message},
            {
                "role": "bot",
                "content": result["reply"],
                "thinking": result.get("thinking") or None,
            },
        ],
        ready_for_prompt=bool(result.get("ready_for_prompt")),
    )
    _bootstraps[session_id] = bootstrap
    return bootstrap


def get_bootstrap(session_id: str) -> BootstrapSession | None:
    _cleanup_expired()
    return _bootstraps.get(session_id)


def bootstrap_to_dict(bootstrap: BootstrapSession) -> dict[str, Any]:
    return {
        "session_id": bootstrap.session_id,
        "title": bootstrap.title,
        "messages": bootstrap.messages,
        "ready_for_prompt": bootstrap.ready_for_prompt,
    }
