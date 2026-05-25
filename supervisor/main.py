import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import threading
import webbrowser

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import SUPERVISOR_HOST, SUPERVISOR_PORT
from supervisor import llm

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Genie Supervisor")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    thinking: str = ""
    ready_for_prompt: bool


class SuggestResponse(BaseModel):
    cursor_prompt: str
    target_mode: str = "plan"


class SeedRequest(BaseModel):
    context: str


@app.on_event("startup")
def startup() -> None:
    llm.ensure_api_key()
    llm.reset_chat()
    print(f"\n[Supervisor] {SUPERVISOR_HOST}:{SUPERVISOR_PORT} — 프롬프트는 복사 후 Cursor에 붙여넣으세요.\n")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="메시지가 비어 있습니다.")

    try:
        result = llm.chat_send(message)
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
