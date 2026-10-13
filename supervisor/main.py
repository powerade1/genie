import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import threading
import webbrowser

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from config import SUPERVISOR_HOST, SUPERVISOR_PORT
from supervisor import cursor_cli, cursor_jobs, llm, sessions

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Genie Supervisor")


class ChatRequest(BaseModel):
    message: str
    cursor_execution_completed: bool = False


class ChatResponse(BaseModel):
    reply: str
    thinking: str = ""
    ready_for_prompt: bool


class SuggestResponse(BaseModel):
    cursor_prompt: str
    target_mode: str = "plan"


class SeedRequest(BaseModel):
    context: str


class CreateSessionRequest(BaseModel):
    message: str


class CreateSessionResponse(BaseModel):
    session_id: str
    url: str
    reply: str
    thinking: str = ""
    ready_for_prompt: bool


class BootstrapResponse(BaseModel):
    session_id: str
    title: str
    messages: list[dict]
    ready_for_prompt: bool


class ExecuteRequest(BaseModel):
    prompt: str
    target_mode: str = "plan"


class ExecuteResponse(BaseModel):
    ok: bool
    message: str
    job_id: str


class JobSnapshotResponse(BaseModel):
    id: str
    target_mode: str
    status: str
    log_text: str
    question: str | None = None
    error: str | None = None
    rounds_count: int = 0
    max_auto_rounds: int = 0
    created_at: str
    finished_at: str | None = None


class JobRespondRequest(BaseModel):
    text: str


class JobRespondResponse(BaseModel):
    ok: bool
    status: str


@app.on_event("startup")
def startup() -> None:
    llm.ensure_api_key()
    llm.reset_chat()
    print(
        f"\n[Supervisor] {SUPERVISOR_HOST}:{SUPERVISOR_PORT} — "
        "프롬프트는 [Cursor 실행] 버튼으로 CLI에 전달하고, 우측 패널에서 진행 로그를 확인하세요.\n"
    )


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="메시지가 비어 있습니다.")

    try:
        result = llm.chat_send(
            message,
            cursor_execution_completed=body.cursor_execution_completed,
        )
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return ChatResponse(**result)


@app.post("/suggest-prompt", response_model=SuggestResponse)
def suggest_prompt():
    try:
        result = llm.suggest_prompt()
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return SuggestResponse(**result)


@app.post("/execute-prompt", response_model=ExecuteResponse)
def execute_prompt(body: ExecuteRequest):
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="프롬프트가 비어 있습니다.")

    try:
        job = cursor_jobs.start_job(prompt, body.target_mode)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    mode_label = cursor_cli.mode_label(body.target_mode)
    return ExecuteResponse(
        ok=True,
        job_id=job.id,
        message=f"{mode_label} 모드로 Cursor CLI 실행을 시작했습니다.",
    )


@app.get("/execute-jobs/{job_id}", response_model=JobSnapshotResponse)
def get_execute_job(job_id: str):
    snapshot = cursor_jobs.get_job_snapshot(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="실행 작업을 찾을 수 없습니다.")
    return JobSnapshotResponse(**snapshot)


@app.get("/execute-jobs/{job_id}/stream")
async def stream_execute_job(job_id: str):
    if cursor_jobs.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="실행 작업을 찾을 수 없습니다.")

    async def event_generator():
        last_log_text = ""
        last_status = ""
        last_question = None
        while True:
            snapshot = cursor_jobs.get_job_snapshot(job_id)
            if snapshot is None:
                yield f"data: {json.dumps({'error': 'not found'})}\n\n"
                return

            log_changed = snapshot["log_text"] != last_log_text
            status_changed = snapshot["status"] != last_status
            question_changed = snapshot.get("question") != last_question

            if log_changed or status_changed or question_changed:
                payload = {
                    "log_text": snapshot["log_text"],
                    "status": snapshot["status"],
                    "question": snapshot.get("question"),
                    "rounds_count": snapshot.get("rounds_count", 0),
                    "max_auto_rounds": snapshot.get("max_auto_rounds", 0),
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                last_log_text = snapshot["log_text"]
                last_status = snapshot["status"]
                last_question = snapshot.get("question")

            if snapshot["status"] in ("completed", "failed"):
                done_payload = {
                    "done": True,
                    "status": snapshot["status"],
                    "log_text": snapshot["log_text"],
                    "question": snapshot.get("question"),
                    "error": snapshot["error"],
                }
                yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
                return

            await asyncio.sleep(0.25)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/execute-jobs/{job_id}/respond", response_model=JobRespondResponse)
def respond_to_job(job_id: str, body: JobRespondRequest):
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="답변이 비어 있습니다.")

    try:
        job = cursor_jobs.respond_to_job(job_id, text)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return JobRespondResponse(ok=True, status=job.status)


@app.post("/sessions/create", response_model=CreateSessionResponse)
def create_session(body: CreateSessionRequest):
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message가 비어 있습니다.")

    try:
        bootstrap = sessions.create_session(message)
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    base_url = f"http://{SUPERVISOR_HOST}:{SUPERVISOR_PORT}"
    bot_message = bootstrap.messages[1]
    return CreateSessionResponse(
        session_id=bootstrap.session_id,
        url=f"{base_url}/?session_id={bootstrap.session_id}",
        reply=bot_message["content"],
        thinking=bot_message.get("thinking") or "",
        ready_for_prompt=bootstrap.ready_for_prompt,
    )


@app.get("/sessions/{session_id}/bootstrap", response_model=BootstrapResponse)
def get_session_bootstrap(session_id: str):
    bootstrap = sessions.get_bootstrap(session_id)
    if bootstrap is None:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    return BootstrapResponse(**sessions.bootstrap_to_dict(bootstrap))


@app.post("/seed")
def seed(body: SeedRequest):
    context = body.context.strip()
    if not context:
        raise HTTPException(status_code=400, detail="context가 비어 있습니다.")
    llm.seed_chat(context)
    return {"ok": True}


@app.post("/reset")
def reset():
    llm.reset_chat()
    return {"ok": True}


def _open_browser(url: str) -> None:
    webbrowser.open(url)


def main() -> None:
    url = f"http://{SUPERVISOR_HOST}:{SUPERVISOR_PORT}"
    print(f"[Supervisor] {url} 에서 실행 중")

    threading.Timer(1.0, _open_browser, args=(url,)).start()

    uvicorn.run(
        app,
        host=SUPERVISOR_HOST,
        port=SUPERVISOR_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
