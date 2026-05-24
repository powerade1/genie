import asyncio
import os
import sys
import time
import traceback

import pyaudio
from google import genai
from google.genai import types

from actions import apps, browser, files, system, web, youtube
from config import (
    CHUNK_SIZE,
    GEMINI_MODEL,
    GEMINI_VOICE,
    IDLE_TIMEOUT_SECONDS,
    RECEIVE_SAMPLE_RATE,
    SEND_SAMPLE_RATE,
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
        "name": "end_session",
        "description": (
            "사용자가 대화를 끝내려는 의도(예: '끝', '그만', '고마워', "
            "'잘 가', '됐어')를 보이면 호출합니다."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
]


def _end_session_action() -> str:
    return "대기 모드로 전환합니다."


ACTION_MAP = {
    "open_app": apps.open_app,
    "close_app": apps.close_app,
    "set_volume": system.set_volume,
    "set_brightness": system.set_brightness,
    "open_url": browser.open_url,
    "file_operation": files.file_operation,
    "search_web": web.search_web,
    "play_youtube": youtube.play_youtube,
    "stop_youtube": youtube.stop_youtube,
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
    # 스피커→마이크 에코로 재응답하지 않도록 재생 후 추가 음소거 시간
    _PLAYBACK_MUTE_TAIL_SEC = 1.0

    def __init__(self):
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
        # True이면 마이크 입력 대신 무음 전송 (스피커 에코 방지)
        self._mute_mic = False

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
            msg = await self.out_queue.get()
            if self._mute_mic and isinstance(msg, dict) and msg.get("data"):
                msg = {
                    "data": b"\x00" * len(msg["data"]),
                    "mime_type": msg.get("mime_type", "audio/pcm"),
                }
            await self.session.send_realtime_input(audio=msg)

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
                        # 함수 호출 turn: 호출 전 음성은 재생하지 않음 (결과 후 한 번만 말하도록)
                        while not self.audio_in_queue.empty():
                            self.audio_in_queue.get_nowait()

                        end_session_requested = False
                        function_responses = []
                        for function_call in response.tool_call.function_calls:
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

                            if func_name == "end_session":
                                end_session_requested = True

                            function_responses.append(
                                types.FunctionResponse(
                                    id=function_call.id,
                                    name=func_name,
                                    response={"result": result},
                                )
                            )

                        if function_responses:
                            await self.session.send_tool_response(function_responses=function_responses)

                        if end_session_requested:
                            print("[end_session] 모델이 종료를 요청했습니다. 마지막 응답까지 재생 후 종료합니다.")
                            self._end_requested = True

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
                            while not self.audio_in_queue.empty():
                                await asyncio.sleep(0.05)
                            await asyncio.sleep(self._PLAYBACK_TAIL_SEC)
                            self.done.set()
                            return
            except asyncio.CancelledError:
                return
            except Exception as error:
                if self.done.is_set():
                    return
                print(f"\n[session] 연결 오류: {error}")
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
            "proactivity": {"proactive_audio": True},
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
                self._mute_mic = False

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
            real_errors = [
                exc for exc in error_group.exceptions
                if not isinstance(exc, asyncio.CancelledError)
            ]
            if real_errors:
                print("\n[session] 작업 중 오류 발생")
                traceback.print_exception(error_group)
        except Exception as error:
            print(f"\n[session] 오류: {error}")
        finally:
            self.cleanup()

    def cleanup(self):
        # 종료 시점에는 백그라운드 write/read 스레드와 경합할 수 있어
        # PortAudio 내부 에러(-9986 등)가 자주 보고된다. 정리는 best-effort.
        for stream_attr in ("audio_stream", "speaker_stream"):
            stream = getattr(self, stream_attr)
            if stream is None:
                continue
            try:
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


async def run_live_session():
    """Wake word 감지 후 Gemini Live API 세션 실행"""
    session = LiveSession()
    await session.run()


def ensure_api_key():
    if not os.environ.get("GEMINI_API_KEY"):
        raise EnvironmentError(
            "GEMINI_API_KEY 환경변수가 설정되지 않았습니다.\n"
            "export GEMINI_API_KEY='your-api-key'"
        )
