import json
import subprocess
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from config import CURSOR_FOLLOWUP_ENABLED, CURSOR_FOLLOWUP_MAX_ROUNDS
from supervisor import llm
from supervisor.cursor_cli import build_agent_command, resolve_workspace

JobStatus = Literal["running", "waiting_for_user", "completed", "failed"]

_MAX_JOBS = 20


@dataclass
class CursorJob:
    id: str
    prompt: str  # 최초 프롬프트
    target_mode: str
    status: JobStatus = "running"
    lines: list[str] = field(default_factory=list)
    log_text: str = ""
    error: str | None = None
    question: str | None = None
    last_assistant_text: str = ""
    rounds_count: int = 0  # 자동 라운드 카운트 (사용자 라운드 제외)
    max_auto_rounds: int = CURSOR_FOLLOWUP_MAX_ROUNDS
    history: list[dict] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: str | None = None
    _assistant_line_index: int | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _rebuild_log_text(self) -> None:
        self.log_text = "\n".join(self.lines)

    def _append_line_unlocked(self, line: str) -> None:
        if line is None:
            return
        self.lines.append(line)
        self._rebuild_log_text()

    def append_line(self, line: str) -> None:
        if not line:
            return
        with self._lock:
            self._append_line_unlocked(line)
            self._assistant_line_index = None

    def set_assistant_text(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            if (
                self._assistant_line_index is not None
                and 0 <= self._assistant_line_index < len(self.lines)
            ):
                self.lines[self._assistant_line_index] = text
            else:
                self.lines.append(text)
                self._assistant_line_index = len(self.lines) - 1
            self._rebuild_log_text()
            self.last_assistant_text = text

    def start_round(self, header: str) -> None:
        with self._lock:
            if self.lines:
                self.lines.append("")
            self.lines.append(header)
            self._assistant_line_index = None
            self.last_assistant_text = ""
            self._rebuild_log_text()

    def set_waiting_for_user(self, question: str, reasoning: str = "") -> None:
        with self._lock:
            self.status = "waiting_for_user"
            self.question = question
            if reasoning:
                self._append_line_unlocked(f"[Supervisor 판단] {reasoning}")
            self._append_line_unlocked(f"[질문 대기] {question}")
            self._assistant_line_index = None
            self.finished_at = None

    def append_history(self, kind: str, text: str) -> None:
        if not text:
            return
        with self._lock:
            self.history.append({"kind": kind, "text": text})

    def finish(self, status: JobStatus, error: str | None = None) -> None:
        with self._lock:
            self.status = status
            self.error = error
            self.finished_at = datetime.now(timezone.utc).isoformat()

    def reset_to_running(self) -> None:
        with self._lock:
            self.status = "running"
            self.question = None
            self._assistant_line_index = None
            self.finished_at = None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "target_mode": self.target_mode,
                "status": self.status,
                "log_text": self.log_text,
                "question": self.question,
                "error": self.error,
                "rounds_count": self.rounds_count,
                "max_auto_rounds": self.max_auto_rounds,
                "created_at": self.created_at,
                "finished_at": self.finished_at,
            }


_jobs: OrderedDict[str, CursorJob] = OrderedDict()
_registry_lock = threading.Lock()


def _format_stream_event(event: dict) -> tuple[str, str | None]:
    """Returns (kind, text) where kind is line|assistant|skip."""
    event_type = event.get("type")

    if event_type == "system" and event.get("subtype") == "init":
        model = event.get("model", "unknown")
        cwd = event.get("cwd", "")
        return "line", f"[시작] {model} · {cwd}"

    if event_type == "assistant":
        content = event.get("message", {}).get("content", [])
        chunks = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = "".join(chunks).strip()
        if text:
            return "assistant", text
        return "skip", None

    if event_type == "tool_call" or event_type == "tool":
        name = event.get("name") or event.get("tool_name") or "tool"
        return "line", f"[도구] {name}"

    if event_type == "result":
        duration_ms = event.get("duration_ms")
        duration = f" · {duration_ms}ms" if duration_ms else ""
        if event.get("is_error"):
            result = event.get("result") or event.get("error") or "실패"
            return "line", f"[오류] {result}{duration}"
        result = event.get("result") or "완료"
        return "line", f"[완료] {result}{duration}"

    return "skip", None


def _consume_stdout(job: CursorJob, process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            job.append_line(line)
            continue

        kind, text = _format_stream_event(event)
        if kind == "line" and text:
            job.append_line(text)
        elif kind == "assistant" and text:
            job.set_assistant_text(text)


def _run_round(
    job: CursorJob,
    prompt: str,
    *,
    continue_session: bool,
    round_label: str,
) -> bool:
    """단일 cursor 호출. 성공이면 True, 실패면 job.finish 호출 후 False."""
    job.start_round(f"────────── {round_label} ──────────")

    try:
        cmd = build_agent_command(
            prompt,
            job.target_mode,
            continue_session=continue_session,
        )
    except (ValueError, FileNotFoundError) as error:
        job.append_line(f"[오류] {error}")
        job.finish("failed", str(error))
        return False

    workspace = str(resolve_workspace())

    try:
        process = subprocess.Popen(
            cmd,
            cwd=workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except OSError as error:
        job.append_line(f"[오류] Cursor CLI 실행 실패: {error}")
        job.finish("failed", str(error))
        return False

    stderr_chunks: list[str] = []

    def read_stderr() -> None:
        assert process.stderr is not None
        for chunk in process.stderr:
            stderr_chunks.append(chunk)

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()
    _consume_stdout(job, process)
    stderr_thread.join(timeout=1)
    return_code = process.wait()

    stderr_text = "".join(stderr_chunks).strip()
    if return_code != 0:
        if stderr_text:
            job.append_line(f"[stderr] {stderr_text}")
        job.finish("failed", stderr_text or f"exit code {return_code}")
        return False

    job.append_history("cursor", job.last_assistant_text)
    return True


def _run_followup_loop(
    job: CursorJob,
    initial_prompt: str,
    *,
    continue_session: bool,
    initial_label: str,
) -> None:
    current_prompt = initial_prompt
    current_continue = continue_session
    current_label = initial_label

    while True:
        ok = _run_round(
            job,
            current_prompt,
            continue_session=current_continue,
            round_label=current_label,
        )
        if not ok:
            return

        if not CURSOR_FOLLOWUP_ENABLED:
            job.finish("completed")
            return

        # plan 모드 외에는 후속 라우팅 불필요
        if job.target_mode != "plan":
            job.finish("completed")
            return

        try:
            followup = llm.analyze_cursor_followup(
                original_prompt=job.prompt,
                target_mode=job.target_mode,
                cursor_output=job.last_assistant_text,
                previous_rounds=job.history,
            )
        except Exception as error:
            job.append_line(f"[Supervisor 분석 실패] {error}")
            job.finish("completed")
            return

        action = followup.get("action") or "finish"
        reasoning = (followup.get("reasoning") or "").strip()

        if action == "finish":
            if reasoning:
                job.append_line(f"[Supervisor] {reasoning}")
            job.finish("completed")
            return

        if action == "ask_user":
            question = (followup.get("question") or "").strip()
            if not question:
                question = (
                    "Cursor가 결정을 요청했지만 자동 답변하기 어렵습니다. "
                    "어떻게 진행할까요?"
                )
            job.set_waiting_for_user(question, reasoning)
            return

        # auto_continue
        answer = (followup.get("auto_answer") or "").strip()
        if not answer:
            job.append_line("[Supervisor] 자동 답변이 비어 있어 종료.")
            job.finish("completed")
            return

        if job.rounds_count >= job.max_auto_rounds:
            suggestion = answer if len(answer) <= 200 else answer[:200] + "…"
            question = (
                f"자동 응답 한도({job.max_auto_rounds}회)에 도달했습니다. "
                f"직접 답변을 입력해 주세요. (Supervisor 제안: {suggestion})"
            )
            job.set_waiting_for_user(question, reasoning)
            return

        job.rounds_count += 1
        job.append_line(f"[자동 응답 #{job.rounds_count}] {answer}")
        if reasoning:
            job.append_line(f"  (근거: {reasoning})")
        job.append_history("auto_answer", answer)

        current_prompt = answer
        current_continue = True
        current_label = f"Cursor 후속 응답 (자동 #{job.rounds_count})"


def _store_job(job: CursorJob) -> None:
    with _registry_lock:
        _jobs[job.id] = job
        while len(_jobs) > _MAX_JOBS:
            _jobs.popitem(last=False)


def start_job(prompt: str, target_mode: str) -> CursorJob:
    mode = target_mode.strip().lower() or "plan"
    # 명령어 빌드를 미리 한 번 호출해서 검증 (잘못된 mode/CLI 경로면 즉시 raise)
    build_agent_command(prompt, mode, continue_session=False)

    job = CursorJob(
        id=str(uuid.uuid4()),
        prompt=prompt,
        target_mode=mode,
    )
    job.append_history("user_prompt", prompt)
    _store_job(job)

    thread = threading.Thread(
        target=_run_followup_loop,
        args=(job, prompt),
        kwargs={"continue_session": False, "initial_label": "Cursor 응답"},
        daemon=True,
    )
    thread.start()
    return job


def respond_to_job(job_id: str, text: str) -> CursorJob:
    job = get_job(job_id)
    if job is None:
        raise KeyError(f"Job not found: {job_id}")
    if job.status != "waiting_for_user":
        raise ValueError(
            f"현재 상태가 waiting_for_user가 아닙니다: {job.status}"
        )
    text = (text or "").strip()
    if not text:
        raise ValueError("답변이 비어 있습니다.")

    job.reset_to_running()
    job.append_line(f"[사용자 답변] {text}")
    job.append_history("user_answer", text)

    thread = threading.Thread(
        target=_run_followup_loop,
        args=(job, text),
        kwargs={
            "continue_session": True,
            "initial_label": "Cursor 후속 응답 (사용자 답변)",
        },
        daemon=True,
    )
    thread.start()
    return job


def get_job(job_id: str) -> CursorJob | None:
    with _registry_lock:
        return _jobs.get(job_id)


def get_job_snapshot(job_id: str) -> dict | None:
    job = get_job(job_id)
    if job is None:
        return None
    return job.snapshot()
