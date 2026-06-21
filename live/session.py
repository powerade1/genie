import asyncio
import contextlib
import os
import sys
import time
import traceback

import pyaudio
from google import genai
from google.genai import types

from actions import apps, browser, custom_actions, files, supervisor, system, web, youtube
from audio.echo_cancel import get_echo_canceller
from audio.speaker import LiveSpeakerGate, get_speaker_verifier
from config import (
    CHUNK_SIZE,
    ECHO_CANCELLATION_ENABLED,
    GEMINI_MODEL,
    GEMINI_VOICE,
    IDLE_TIMEOUT_SECONDS,
    LIVE_SPEAKER_GATE_ENABLED,
    LIVE_SPEAKER_GATE_HOLD_SEC,
    RECEIVE_SAMPLE_RATE,
    SEND_SAMPLE_RATE,
    SPEAKER_VERIFICATION_ENABLED,
    SYSTEM_PROMPT,
    VAD_END_OF_SPEECH_SENSITIVITY,
    VAD_SILENCE_DURATION_MS,
    VAD_START_OF_SPEECH_SENSITIVITY,
)

if sys.version_info < (3, 11, 0):
    import exceptiongroup
    import taskgroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1


def _print_session_error(error: BaseException, prefix: str = "오류") -> None:
    """Live API 세션 에러를 사용자에게 의미있게 출력한다.

    1008/1011 등 모호한 close 코드 + ``credit`` / ``quota`` / ``billing`` 같은
    힌트가 메시지에 보이면 결제 만료 가능성을 함께 안내한다.
    """
    text = str(error)
    print(f"\n[session] {prefix}: {text}", flush=True)

    lower = text.lower()
    hint = None
    if any(token in lower for token in ("credit", "depleted", "billing", "prepayment")):
        hint = (
            "  → Google AI Studio 결제 잔액이 부족할 수 있습니다. "
            "https://ai.studio/projects 에서 결제를 갱신하세요."
        )
    elif "quota" in lower or "rate" in lower:
        hint = "  → API quota 또는 rate limit 에 걸렸을 수 있습니다."
    elif "1008" in text and "implemented" in lower:
        # 결제 만료가 1008 으로 위장되어 오는 경우가 흔하다.
        hint = (
            "  → 1008 close 는 보통 결제 잔액 만료 또는 모델/리전 미지원이 원인입니다. "
            "AI Studio 결제 상태를 먼저 확인하세요: https://ai.studio/projects"
        )
    elif "permission" in lower or "unauthorized" in lower or "401" in text or "403" in text:
        hint = "  → API 키 권한 또는 만료를 확인하세요 (.env 의 GEMINI_API_KEY)."

    if hint:
        print(hint, flush=True)


def _is_graceful_close_error(error: BaseException) -> bool:
    """의도적/정상 세션 종료 직후 서버가 보내는 close 를 구분한다."""
    text = str(error).lower()
    return any(
        token in text
        for token in (
            "1000",
            "1001",
            "1008",
            "closed",
            "cancelled",
            "connection reset",
            "not connected",
        )
    )


@contextlib.contextmanager
def _suppress_native_streams():
    """C 레벨 fprintf(stdout/stderr) 출력을 잠깐 가린다.

    종료 시 PortAudio 가 ``||PaMacCore (AUHAL)|| Error on line 2796: err='-50'``
    같은 메시지를 직접 출력하는데, macOS 빌드에서는 의외로 **stdout** 으로 가는
    경우가 있어서 fd 1, 2 를 모두 가려야 안전하다 (Python `print` 와는 무관한
    fd redirection). cleanup 구간에서만 사용한다.
    """
    devnulls: dict[int, int | None] = {1: None, 2: None}
    saved: dict[int, int | None] = {1: None, 2: None}
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass
    try:
        for fd in (1, 2):
            try:
                devnulls[fd] = os.open(os.devnull, os.O_WRONLY)
                saved[fd] = os.dup(fd)
                os.dup2(devnulls[fd], fd)
            except OSError:
                pass
        yield
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass
        for fd in (1, 2):
            saved_fd = saved[fd]
            if saved_fd is not None:
                try:
                    os.dup2(saved_fd, fd)
                except OSError:
                    pass
                try:
                    os.close(saved_fd)
                except OSError:
                    pass
            dn = devnulls[fd]
            if dn is not None:
                try:
                    os.close(dn)
                except OSError:
                    pass

TOOLS = [
    {
        "name": "open_app",
        "description": "맥북에서 애플리케이션을 실행합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": (
                        "macOS 영어 앱 이름 (예: KakaoTalk, Google Chrome, Notion, Spotify). "
                        "한국어 이름은 사용하지 마세요."
                    ),
                }
            },
            "required": ["app_name"],
        },
    },
    {
        "name": "close_app",
        "description": "실행 중인 애플리케이션을 종료합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {"type": "string", "description": "종료할 앱 이름"}
            },
            "required": ["app_name"],
        },
    },
    {
        "name": "set_volume",
        "description": "시스템 볼륨을 조절합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "level": {"type": "integer", "description": "볼륨 레벨 0~100"},
                "direction": {"type": "string", "enum": ["up", "down", "mute"]},
            },
        },
    },
    {
        "name": "set_brightness",
        "description": "화면 밝기를 조절합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "level": {"type": "integer", "description": "밝기 레벨 0~100"},
                "direction": {"type": "string", "enum": ["up", "down"]},
            },
        },
    },
    {
        "name": "open_url",
        "description": "브라우저에서 URL 또는 검색어를 엽니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "URL 또는 검색어"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "file_operation",
        "description": "파일을 열거나 폴더를 탐색합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["open", "reveal", "trash"]},
                "path": {"type": "string", "description": "파일 또는 폴더 경로"},
            },
            "required": ["action", "path"],
        },
    },
    {
        "name": "search_web",
        "description": (
            "최신 정보, 뉴스, 시세, 날짜/시간이 중요한 사실, 또는 모델이 학습한 시점 "
            "이후의 정보가 필요한 질문에서 호출합니다. 인터넷 검색 결과를 받아 "
            "자연어로 요약해 답변하세요."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색어 (한국어 또는 영어)"},
                "max_results": {
                    "type": "integer",
                    "description": "가져올 결과 수 (기본 5, 최대 10)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "start_work_routine",
        "description": (
            "업무·작업 시작 루틴을 실행합니다. "
            "'업무 시작', '작업 시작', '루틴 시작', '일 시작', '개발 시작', "
            "'준비해줘', '세팅해줘' 등 업무 환경을 한꺼번에 준비해 달라는 요청에 사용하세요. "
            "종팝 플레이리스트를 재생한 뒤 Cursor, Google Chrome, Notion을 순서대로 실행합니다. "
            "앱 하나만 열거나 음악만 재생할 때는 open_app이나 play_youtube를 사용하세요."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "play_youtube",
        "description": (
            "YouTube에서 키워드로 영상·음악·플레이리스트를 검색해 재생합니다. "
            "'틀어줘', '플리', '노래', '유튜브' 요청에 사용. "
            "open_url 대신 이 함수를 사용하세요. "
            "query는 사용자가 말한 검색어를 한국어 원문 그대로 전달하세요. "
            "음성 유사어로 영어 번역하지 마세요 (예: '종팝' → 'J-POP' 금지)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "사용자가 말한 검색어 원문 (한국어 그대로). "
                        "예: 종팝 플레이리스트, 잔잔한 재즈 플리, 아이유 노래. "
                        "'종팝'을 'J-POP'으로 바꾸지 마세요."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "stop_youtube",
        "description": "YouTube 재생을 중지합니다.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "create_tool",
        "description": (
            "코딩·개발·아키텍처 작업을 Supervisor 웹 UI에서 시작할 때 호출합니다. "
            "사용자의 음성 요청을 tool_description으로 요약해 전달하세요."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool_description": {
                    "type": "string",
                    "description": "Supervisor에 전달할 작업 설명 (사용자 요청 요약)",
                }
            },
            "required": ["tool_description"],
        },
    },
    {
        "name": "end_session",
        "description": (
            "사용자가 대화를 끝내려는 의도(예: '끝', '그만', '마무리', '고마워', "
            "'됐어')를 보이면, 마지막 인사를 말한 **후** 호출합니다. "
            "인사와 동시에 호출하지 마세요."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
]


def _end_session_action() -> str:
    return "대기 모드로 전환합니다."


ACTION_MAP = {
    "start_work_routine": custom_actions.start_work_routine,
    "open_app": apps.open_app,
    "close_app": apps.close_app,
    "set_volume": system.set_volume,
    "set_brightness": system.set_brightness,
    "open_url": browser.open_url,
    "file_operation": files.file_operation,
    "search_web": web.search_web,
    "play_youtube": youtube.play_youtube,
    "stop_youtube": youtube.stop_youtube,
    "create_tool": supervisor.create_tool,
    "end_session": _end_session_action,
}

_client = None


def get_client():
    global _client
    if _client is None:
        _client = genai.Client(http_options={"api_version": "v1alpha"})
    return _client


class LiveSession:
    # 마지막 청크가 스피커 버퍼에서 빠져나가는 짧은 여유 시간
    _PLAYBACK_TAIL_SEC = 0.3
    # end_session 마지막 인사가 스피커 버퍼에서 완전히 나갈 때까지 추가 대기
    _END_SESSION_PLAYBACK_TAIL_SEC = 1.2
    # end_session 재생 drain 최대 대기 (초)
    _END_SESSION_DRAIN_TIMEOUT_SEC = 20.0
    # tool_call 이 인사 오디오보다 먼저 올 때 남은 청크를 받을 최소 대기
    _END_SESSION_MIN_WAIT_SEC = 1.5
    # 스피커→마이크 에코로 재응답하지 않도록 재생 후 추가 음소거 시간
    _PLAYBACK_MUTE_TAIL_SEC = 1.0

    def __init__(self, *, speaker_gate_warmup: bool = False):
        self.pya = pyaudio.PyAudio()
        self.session = None
        self.audio_stream = None
        self.speaker_stream = None
        self.audio_in_queue = None
        self.out_queue = None
        self.done = asyncio.Event()
        # 모델이 응답을 마친 직후 시점부터 idle 카운트 시작
        self._turn_complete_event = asyncio.Event()
        self._last_activity_ts = time.monotonic()
        self._end_requested = False
        self._end_finish_started = False
        self._end_requested_at = 0.0
        # True이면 마이크 입력 대신 무음 전송 (스피커 에코 방지)
        self._mute_mic = False
        self._speaker_gate: LiveSpeakerGate | None = None
        self._speaker_gate_warmup = speaker_gate_warmup
        self._echo = get_echo_canceller()

    async def _wait_for_playback_drain(
        self,
        *,
        tail_sec: float | None = None,
        timeout_sec: float | None = None,
    ) -> None:
        """스피커 큐가 비워질 때까지 기다린다 (end_session 마지막 인사용)."""
        tail = self._END_SESSION_PLAYBACK_TAIL_SEC if tail_sec is None else tail_sec
        timeout = (
            self._END_SESSION_DRAIN_TIMEOUT_SEC
            if timeout_sec is None
            else timeout_sec
        )
        deadline = time.monotonic() + timeout
        empty_since: float | None = None
        while time.monotonic() < deadline:
            if self.audio_in_queue is None or self.audio_in_queue.empty():
                if empty_since is None:
                    empty_since = time.monotonic()
                elif time.monotonic() - empty_since >= 0.6:
                    await asyncio.sleep(self._PLAYBACK_MUTE_TAIL_SEC + tail)
                    return
            else:
                empty_since = None
            await asyncio.sleep(0.05)

    async def _finish_end_session(self) -> None:
        """end_session: 마지막 음성 재생 후 조용히 세션을 닫는다."""
        if self.done.is_set() or self._end_finish_started:
            return
        self._end_finish_started = True
        try:
            if self._end_requested_at:
                remaining = self._END_SESSION_MIN_WAIT_SEC - (
                    time.monotonic() - self._end_requested_at
                )
                if remaining > 0:
                    await asyncio.sleep(remaining)
            await self._wait_for_playback_drain()
        finally:
            self.done.set()

    async def listen_audio(self):
        mic_info = self.pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )

        while not self.done.is_set():
            try:
                data = await asyncio.to_thread(
                    self.audio_stream.read,
                    CHUNK_SIZE,
                    exception_on_overflow=False,
                )
            except OSError as error:
                if not self.done.is_set():
                    print(f"\n[audio] 마이크 오류: {error}")
                    self.done.set()
                return
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def send_realtime(self):
        while not self.done.is_set():
            try:
                msg = await self.out_queue.get()
            except asyncio.CancelledError:
                return
            if self.done.is_set():
                return
            if self._mute_mic and isinstance(msg, dict) and msg.get("data"):
                msg = {
                    "data": b"\x00" * len(msg["data"]),
                    "mime_type": msg.get("mime_type", "audio/pcm"),
                }
            elif (
                ECHO_CANCELLATION_ENABLED
                and self._echo.enabled
                and isinstance(msg, dict)
                and msg.get("data")
            ):
                msg = {
                    "data": self._echo.process_pcm16(msg["data"]),
                    "mime_type": msg.get("mime_type", "audio/pcm"),
                }
            elif (
                self._speaker_gate is not None
                and self._speaker_gate.enabled
                and isinstance(msg, dict)
                and msg.get("data")
            ):
                filtered = self._speaker_gate.filter_chunk(msg["data"])
                if filtered != msg["data"]:
                    msg = {
                        "data": filtered,
                        "mime_type": msg.get("mime_type", "audio/pcm"),
                    }
            try:
                await self.session.send_realtime_input(audio=msg)
            except asyncio.CancelledError:
                return
            except Exception:
                if self.done.is_set() or self._end_requested:
                    return
                raise

    def _extract_audio(self, response) -> bytes | None:
        server_content = response.server_content
        if not server_content or not server_content.model_turn:
            return None

        chunks = []
        for part in server_content.model_turn.parts or []:
            if part.inline_data and isinstance(part.inline_data.data, bytes):
                chunks.append(part.inline_data.data)

        return b"".join(chunks) if chunks else None

    async def receive_audio(self):
        while not self.done.is_set():
            try:
                turn = self.session.receive()
                async for response in turn:
                    if self.done.is_set():
                        return

                    audio_data = self._extract_audio(response)
                    if audio_data:
                        self._mute_mic = True
                        self.audio_in_queue.put_nowait(audio_data)
                        # 모델이 말하는 동안은 idle 아님
                        self._turn_complete_event.clear()
                        self._last_activity_ts = time.monotonic()

                    if response.tool_call:
                        function_calls = list(response.tool_call.function_calls)
                        end_session_requested = any(
                            fc.name == "end_session" for fc in function_calls
                        )
                        if end_session_requested:
                            # send_tool_response 를 end_session 에 보내면 서버가 즉시
                            # WebSocket 을 닫아(1008) 마지막 인사 음성이 끊긴다.
                            # 플래그만 세우고 tool response 는 보내지 않는다.
                            self._end_requested = True
                            self._end_requested_at = time.monotonic()

                        # end_session 단독 호출이면 그 직전 발화("알겠습니다" 등)가
                        # 마지막 인사 그 자체이므로 큐를 비우지 않는다.
                        end_session_only = (
                            len(function_calls) == 1
                            and function_calls[0].name == "end_session"
                        )
                        if not end_session_only:
                            while not self.audio_in_queue.empty():
                                self.audio_in_queue.get_nowait()

                        function_responses = []
                        for function_call in function_calls:
                            func_name = function_call.name
                            args = dict(function_call.args or {})
                            print(f"\n[실행]: {func_name}({args})")

                            if func_name in ACTION_MAP:
                                try:
                                    result = await asyncio.to_thread(
                                        ACTION_MAP[func_name], **args
                                    )
                                except Exception as error:
                                    result = f"실행 오류: {error}"
                                    print(result)
                            else:
                                result = f"알 수 없는 함수: {func_name}"

                            # end_session 은 서버에 응답을 보내지 않는다 (위 주석).
                            if func_name == "end_session":
                                continue

                            function_responses.append(
                                types.FunctionResponse(
                                    id=function_call.id,
                                    name=func_name,
                                    response={"result": result},
                                )
                            )

                        if function_responses:
                            await self.session.send_tool_response(
                                function_responses=function_responses
                            )

                        if end_session_requested:
                            print(
                                "[end_session] 마지막 응답 재생 후 종료합니다 "
                                "(tool response 미전송)."
                            )
                            # receive 루프는 계속 돌며 남은 오디오 청크를 받는다.
                            # 백그라운드에서 큐 drain 후 done 을 세트한다.
                            asyncio.create_task(self._finish_end_session())

                    server_content = response.server_content
                    if not server_content:
                        continue

                    if server_content.interrupted:
                        print(f"[interrupted] 잔여 청크 폐기: {self.audio_in_queue.qsize()}")
                        while not self.audio_in_queue.empty():
                            self.audio_in_queue.get_nowait()
                        self._turn_complete_event.clear()
                        self._last_activity_ts = time.monotonic()
                        continue

                    if server_content.generation_complete:
                        self._last_activity_ts = time.monotonic()

                    if server_content.turn_complete:
                        self._turn_complete_event.set()
                        self._last_activity_ts = time.monotonic()

                        if self._end_requested:
                            await self._finish_end_session()
                            return
            except asyncio.CancelledError:
                return
            except Exception as error:
                if self._end_requested:
                    await self._finish_end_session()
                    return
                if self.done.is_set() or _is_graceful_close_error(error):
                    self.done.set()
                    return
                _print_session_error(error, prefix="연결 오류")
                self.done.set()
                return

    async def play_audio(self):
        self.speaker_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )

        while not self.done.is_set():
            try:
                bytestream = await asyncio.wait_for(self.audio_in_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                if self._mute_mic and self.audio_in_queue.empty():
                    await asyncio.sleep(self._PLAYBACK_MUTE_TAIL_SEC)
                    if self.audio_in_queue.empty():
                        self._mute_mic = False
                continue
            self._mute_mic = True
            if ECHO_CANCELLATION_ENABLED and self._echo.enabled:
                self._echo.add_reference_pcm16_24k(bytestream)
            try:
                await asyncio.to_thread(self.speaker_stream.write, bytestream)
            except OSError as error:
                print(f"\n[audio] 재생 오류: {error}")
                self.done.set()
                return

    async def monitor_idle(self):
        while not self.done.is_set():
            await asyncio.sleep(1)
            if not self._turn_complete_event.is_set():
                continue
            if not self.audio_in_queue.empty():
                continue
            elapsed = time.monotonic() - self._last_activity_ts
            if elapsed >= IDLE_TIMEOUT_SECONDS:
                print(f"\n[idle] {IDLE_TIMEOUT_SECONDS}초 무응답, 대기 모드로 전환")
                self.done.set()
                return

    async def run(self):
        config = {
            "system_instruction": SYSTEM_PROMPT,
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": GEMINI_VOICE}
                }
            },
            "realtime_input_config": {
                "automatic_activity_detection": {
                    "start_of_speech_sensitivity": VAD_START_OF_SPEECH_SENSITIVITY,
                    "end_of_speech_sensitivity": VAD_END_OF_SPEECH_SENSITIVITY,
                    "silence_duration_ms": VAD_SILENCE_DURATION_MS,
                },
                "activity_handling": "NO_INTERRUPTION",
            },
            # 침묵·무관한 입력에 불필요하게 다시 말하지 않도록
            "proactivity": {"proactive_audio": False},
            "tools": [{"function_declarations": TOOLS}],
        }

        try:
            async with (
                get_client().aio.live.connect(model=GEMINI_MODEL, config=config) as session,
                asyncio.TaskGroup() as task_group,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                self.done.clear()
                self._turn_complete_event.clear()
                self._last_activity_ts = time.monotonic()
                self._end_requested = False
                self._end_finish_started = False
                self._end_requested_at = 0.0
                self._mute_mic = False
                if ECHO_CANCELLATION_ENABLED and self._echo.enabled:
                    self._echo.reset()
                if SPEAKER_VERIFICATION_ENABLED and LIVE_SPEAKER_GATE_ENABLED:
                    verifier = get_speaker_verifier()
                    if verifier.is_ready:
                        self._speaker_gate = LiveSpeakerGate(verifier, CHUNK_SIZE)
                        if self._speaker_gate_warmup:
                            self._speaker_gate.force_open(LIVE_SPEAKER_GATE_HOLD_SEC)
                    else:
                        self._speaker_gate = None
                else:
                    self._speaker_gate = None

                print("Gemini Live 연결됨. 말씀하세요.")

                task_group.create_task(self.send_realtime())
                task_group.create_task(self.listen_audio())
                task_group.create_task(self.receive_audio())
                task_group.create_task(self.play_audio())
                task_group.create_task(self.monitor_idle())

                await self.done.wait()
                raise asyncio.CancelledError()
        except asyncio.CancelledError:
            pass
        except BaseExceptionGroup as error_group:
            if self._end_requested:
                pass
            else:
                real_errors = [
                    exc for exc in error_group.exceptions
                    if not isinstance(exc, asyncio.CancelledError)
                    and not _is_graceful_close_error(exc)
                ]
                if real_errors:
                    print("\n[session] 작업 중 오류 발생")
                    traceback.print_exception(error_group)
        except Exception as error:
            if not self._end_requested and not _is_graceful_close_error(error):
                _print_session_error(error, prefix="오류")
        finally:
            self.cleanup()

    def cleanup(self):
        # 종료 시점에는 백그라운드 write/read 스레드와 경합할 수 있어
        # PortAudio 내부 에러(-9986 / Core Audio -50 등)가 종종 stderr 로
        # 직접 출력된다. 정리는 best-effort 이고, 그 잡음만 잠깐 차단한다.
        with _suppress_native_streams():
            for stream_attr in ("audio_stream", "speaker_stream"):
                stream = getattr(self, stream_attr)
                if stream is None:
                    continue
                try:
                    if stream.is_active():
                        stream.stop_stream()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
                setattr(self, stream_attr, None)

            try:
                self.pya.terminate()
            except Exception:
                pass


async def run_live_session(*, speaker_gate_warmup: bool = False):
    """Wake word 감지 후 Gemini Live API 세션 실행"""
    session = LiveSession(speaker_gate_warmup=speaker_gate_warmup)
    await session.run()


def ensure_api_key():
    if not os.environ.get("GEMINI_API_KEY"):
        raise EnvironmentError(
            "GEMINI_API_KEY 환경변수가 설정되지 않았습니다.\n"
            "export GEMINI_API_KEY='your-api-key'"
        )
