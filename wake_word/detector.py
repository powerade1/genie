import threading
import time
from collections import deque

import numpy as np
import pyaudio
from openwakeword.model import Model

from audio.feedback import play_activation_chime
from audio.speaker import get_speaker_verifier
from audio.speech_trim import trim_silence
from config import (
    SPEAKER_VERIFICATION_ENABLED,
    WAKE_EVENT_LOG_ENABLED,
    WAKE_WORD_CONSECUTIVE_FRAMES,
    WAKE_WORD_FRAME_SIZE,
    WAKE_WORD_MODEL_PATH,
    WAKE_WORD_POST_SESSION_COOLDOWN_SEC,
    WAKE_WORD_REFRACTORY_SEC,
    WAKE_WORD_THRESHOLD,
)
from wake_word.event_log import get_wake_event_logger


# macOS가 잠자기에서 깨어난 직후 PortAudio 입력 스트림이 무한 무음만 반환하거나
# OSError를 던지는 경우가 있어, 자동으로 stream을 재오픈하기 위한 임계값들.
_SILENCE_RESTART_SEC = 8.0  # 이 시간 이상 모든 샘플이 0이면 sleep/wake로 간주하고 재시작
_READ_BACKOFF_SEC = 0.5     # read 에러 후 재시도 대기
_RING_SECONDS = 2.0         # 화자 검증용 롤링 버퍼 길이
_POST_CAPTURE_SECONDS = 1.0 # 이벤트 로그: 탐지 직후 추가 녹음


class WakeWordDetector:
    def __init__(
        self,
        model_path: str = WAKE_WORD_MODEL_PATH,
        threshold: float = WAKE_WORD_THRESHOLD,
    ):
        self.model_path = model_path
        self.threshold = threshold
        self.model = Model(wakeword_models=[model_path])
        self.audio: pyaudio.PyAudio | None = None
        self._stream = None
        self._stream_lock = threading.Lock()
        self._restart_requested = threading.Event()
        self._stop_requested = threading.Event()
        self._cooldown_until = 0.0
        self._suspended = False
        self._speaker = get_speaker_verifier() if SPEAKER_VERIFICATION_ENABLED else None
        self._event_logger = get_wake_event_logger()
        self._ring: deque[np.ndarray] = deque()
        self._ring_max_samples = int(16000 * _RING_SECONDS)

    def _append_ring(self, audio_data: np.ndarray) -> None:
        self._ring.append(audio_data.copy())
        total = sum(len(chunk) for chunk in self._ring)
        while total > self._ring_max_samples and self._ring:
            total -= len(self._ring.popleft())

    def _verify_speaker(self) -> tuple[bool, float | None]:
        if self._speaker is None or not self._speaker.is_ready:
            return True, None
        if not self._ring:
            return False, None
        audio = np.concatenate(list(self._ring))
        audio_f = trim_silence(audio)
        if len(audio_f) < 8000:
            print("[wake-word] 음성 구간 너무 짧음, 호출 무시")
            return False, None
        ok, score = self._speaker.is_user(audio_f, sample_rate=16000)
        if not ok:
            print(f"[wake-word] 화자 불일치 (similarity={score:.2f}), 호출 무시")
        return ok, score

    def _capture_event_clip(self, *, include_post: bool = True) -> np.ndarray:
        """탐지 시점 기준 직전 ~2초 + (선택) 직후 1초 오디오."""
        pre = np.concatenate(list(self._ring)) if self._ring else np.array([], dtype=np.int16)
        if not include_post:
            return pre

        post_parts: list[np.ndarray] = []
        post_target = int(16000 * _POST_CAPTURE_SECONDS)
        post_samples = 0

        if self._stream is not None:
            while post_samples < post_target and not self._stop_requested.is_set():
                try:
                    chunk = self._stream.read(
                        WAKE_WORD_FRAME_SIZE,
                        exception_on_overflow=False,
                    )
                except Exception:
                    break
                part = np.frombuffer(chunk, dtype=np.int16)
                post_parts.append(part)
                post_samples += len(part)

        if post_parts:
            post = np.concatenate(post_parts)[:post_target]
            return np.concatenate([pre, post])
        return pre

    def _reset_model(self) -> None:
        try:
            self.model.reset()
        except Exception:
            pass

    def suspend_for_session(self) -> None:
        """Live 세션이 마이크를 쓰는 동안 wake word 입력을 닫는다.

        listen 루프가 callback 동안 read 하지 않으면 PyAudio 버퍼가 쌓여,
        세션 종료 직후 오탐·연속 호출이 폭발한다.
        """
        self._suspended = True
        with self._stream_lock:
            self._close_stream_locked()
        self._reset_model()

    def resume_after_session(self) -> None:
        """세션 종료 후 스트림을 다시 열고, 쌓인/잔여 오디오를 버린 뒤 쿨다운."""
        self._suspended = False
        # LiveSession 이 PyAudio 를 terminate 한 직후 macOS 에서 마이크 핸들이
        # 풀리기까지 짧게 기다린다.
        time.sleep(0.3)
        with self._stream_lock:
            if self.audio is None:
                self.audio = pyaudio.PyAudio()
            self._close_stream_locked()
            self._stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=WAKE_WORD_FRAME_SIZE,
            )
        self._flush_stream(WAKE_WORD_POST_SESSION_COOLDOWN_SEC)
        self._reset_model()
        self._cooldown_until = time.monotonic() + WAKE_WORD_POST_SESSION_COOLDOWN_SEC

    def _flush_stream(self, duration_sec: float) -> None:
        """마이크 버퍼·스피커 잔향 등을 read 만 하며 버린다."""
        if self._stream is None:
            return
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline and not self._stop_requested.is_set():
            try:
                self._stream.read(
                    WAKE_WORD_FRAME_SIZE,
                    exception_on_overflow=False,
                )
            except Exception:
                break
        self._reset_model()

    # --- 외부 제어 -------------------------------------------------------

    def request_restart(self, reason: str = "external") -> None:
        """다음 read 시점에 audio stream을 강제로 재오픈한다.

        macOS sleep/wake 모니터처럼 외부에서 호출해 즉시 복구를 트리거한다.
        """
        print(f"[wake-word] stream 재시작 요청 ({reason})")
        self._restart_requested.set()

    # --- 내부 stream 라이프사이클 ----------------------------------------

    def _open_stream(self) -> None:
        with self._stream_lock:
            self._close_stream_locked()
            if self.audio is None:
                self.audio = pyaudio.PyAudio()
            self._stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=WAKE_WORD_FRAME_SIZE,
            )

    def _close_stream_locked(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _reopen_stream(self, reason: str) -> bool:
        """sleep/wake·OSError 등으로 stream을 강제 재오픈한다. 성공 여부 반환."""
        print(f"[wake-word] audio stream 재오픈 시도: {reason}")
        try:
            with self._stream_lock:
                self._close_stream_locked()
                # PyAudio 인스턴스도 통째로 재생성: 일부 경우 PortAudio
                # 호스트 API 핸들이 깨어난 뒤에도 stale 상태로 남기 때문.
                if self.audio is not None:
                    try:
                        self.audio.terminate()
                    except Exception:
                        pass
                    self.audio = None
                self.audio = pyaudio.PyAudio()
                self._stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=WAKE_WORD_FRAME_SIZE,
                )
            self._reset_model()
            print("[wake-word] audio stream 재오픈 성공")
            return True
        except Exception as error:
            print(f"[wake-word] audio stream 재오픈 실패: {error}")
            return False

    # --- 메인 루프 -------------------------------------------------------

    def listen(self, callback):
        """Wake word 감지 시 callback 실행. 내부에서 stream 자동 복구."""
        self._stop_requested.clear()
        self._restart_requested.clear()
        self._open_stream()

        last_nonzero_time = time.monotonic()
        consecutive_hits = 0
        last_trigger_time = 0.0

        try:
            while not self._stop_requested.is_set():
                if self._suspended:
                    time.sleep(_READ_BACKOFF_SEC)
                    continue

                if self._stream is None:
                    time.sleep(_READ_BACKOFF_SEC)
                    continue

                if self._restart_requested.is_set():
                    self._restart_requested.clear()
                    if not self._reopen_stream("external request"):
                        time.sleep(_READ_BACKOFF_SEC)
                        continue
                    last_nonzero_time = time.monotonic()
                    consecutive_hits = 0

                now = time.monotonic()
                if now < self._cooldown_until:
                    consecutive_hits = 0
                    try:
                        self._stream.read(
                            WAKE_WORD_FRAME_SIZE,
                            exception_on_overflow=False,
                        )
                    except Exception:
                        pass
                    continue

                try:
                    audio_chunk = self._stream.read(
                        WAKE_WORD_FRAME_SIZE,
                        exception_on_overflow=False,
                    )
                except OSError as error:
                    if self._suspended:
                        time.sleep(_READ_BACKOFF_SEC)
                        continue
                    print(f"[wake-word] read OSError: {error}")
                    time.sleep(_READ_BACKOFF_SEC)
                    self._reopen_stream("OSError")
                    last_nonzero_time = time.monotonic()
                    consecutive_hits = 0
                    continue
                except Exception as error:
                    print(f"[wake-word] read 예외: {error}")
                    time.sleep(_READ_BACKOFF_SEC)
                    self._reopen_stream("exception")
                    last_nonzero_time = time.monotonic()
                    consecutive_hits = 0
                    continue

                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                self._append_ring(audio_data)

                if audio_data.any():
                    last_nonzero_time = time.monotonic()
                elif time.monotonic() - last_nonzero_time > _SILENCE_RESTART_SEC:
                    self._reopen_stream("silence timeout")
                    last_nonzero_time = time.monotonic()
                    consecutive_hits = 0
                    continue

                try:
                    predictions = self.model.predict(audio_data)
                except Exception as error:
                    print(f"[wake-word] model.predict 예외: {error}")
                    continue

                best_score = max(predictions.values()) if predictions else 0.0
                if best_score > self.threshold:
                    consecutive_hits += 1
                else:
                    consecutive_hits = 0

                if consecutive_hits < WAKE_WORD_CONSECUTIVE_FRAMES:
                    continue

                if time.monotonic() - last_trigger_time < WAKE_WORD_REFRACTORY_SEC:
                    consecutive_hits = 0
                    continue

                speaker_ok, speaker_score = self._verify_speaker()
                outcome = "accepted" if speaker_ok else "rejected_speaker"

                if WAKE_EVENT_LOG_ENABLED:
                    event_audio = None
                    if speaker_ok:
                        event_audio = self._capture_event_clip(include_post=True)
                    self._event_logger.log_detection(
                        audio=event_audio,
                        predictions=predictions,
                        outcome=outcome,
                        speaker_similarity=speaker_score,
                        threshold=self.threshold,
                    )

                if not speaker_ok:
                    consecutive_hits = 0
                    continue

                print(f"Wake word 감지! (score: {best_score:.2f})")
                last_trigger_time = time.monotonic()
                consecutive_hits = 0
                play_activation_chime()
                callback()
                last_nonzero_time = time.monotonic()
        finally:
            with self._stream_lock:
                self._close_stream_locked()

    def close(self):
        self._stop_requested.set()
        with self._stream_lock:
            self._close_stream_locked()
            if self.audio is not None:
                try:
                    self.audio.terminate()
                except Exception:
                    pass
                self.audio = None
