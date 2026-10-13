import json
import socket
import urllib.error
import urllib.request

from actions import browser
from config import SUPERVISOR_HOST, SUPERVISOR_PORT

_CREATE_TIMEOUT_SEC = 120


def _supervisor_base_url() -> str:
    return f"http://{SUPERVISOR_HOST}:{SUPERVISOR_PORT}"


def _parse_error_body(body: bytes) -> str:
    try:
        data = json.loads(body.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body.decode(errors="replace") or "알 수 없는 오류"
    detail = data.get("detail")
    if isinstance(detail, list):
        return ", ".join(str(item.get("msg", item)) for item in detail)
    return str(detail or data)


def create_tool(tool_description: str) -> str:
    """Supervisor에 새 세션을 만들고 웹 UI를 연다."""
    description = (tool_description or "").strip()
    if not description:
        return "Supervisor에 전달할 작업 설명이 비어 있습니다."

    url = f"{_supervisor_base_url()}/sessions/create"
    payload = json.dumps({"message": description}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=_CREATE_TIMEOUT_SEC) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = _parse_error_body(error.read())
        return f"Supervisor 처리 중 오류가 발생했습니다: {detail}"
    except urllib.error.URLError as error:
        reason = error.reason
        if isinstance(reason, ConnectionRefusedError) or (
            isinstance(reason, OSError) and reason.errno in (61, 111)
        ):
            return (
                "Supervisor 서버가 실행 중이 아닙니다. "
                "python supervisor/main.py 로 서버를 시작해 주세요."
            )
        if isinstance(reason, socket.timeout):
            return "Supervisor 응답 시간이 초과되었습니다."
        return f"Supervisor 연결 실패: {reason}"
    except TimeoutError:
        return "Supervisor 응답 시간이 초과되었습니다."
    except Exception as error:
        return f"Supervisor 연동 오류: {error}"

    session_url = data.get("url")
    if not session_url:
        return "Supervisor 세션 URL을 받지 못했습니다."

    browser.open_url(session_url)
    reply = (data.get("reply") or "").strip()
    if reply:
        summary = reply if len(reply) <= 80 else reply[:77] + "…"
        return f"Supervisor에 새 작업을 등록했습니다. {summary}"
    return "Supervisor에 새 작업을 등록하고 웹 UI를 열었습니다."
