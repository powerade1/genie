"""NLMS 기반 에코 캔슬 (스피커 출력 → 마이크 에코 제거).

Live 세션에서 Gemini 음성(24kHz)을 16kHz 참조 신호로 변환해
마이크 입력에서 재생 잔향을 줄인다. 재생 중에는 session 의 mute 와 병행한다.
"""

from __future__ import annotations

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

from config import (
    ECHO_CANCELLATION_ENABLED,
    ECHO_FILTER_LENGTH,
    ECHO_NLMS_STEP,
    ECHO_REF_DELAY_MS,
    RECEIVE_SAMPLE_RATE,
    SEND_SAMPLE_RATE,
)


class EchoCanceller:
    """Sample-domain NLMS adaptive echo canceller (16 kHz mic / ref)."""

    def __init__(
        self,
        *,
        sample_rate: int = SEND_SAMPLE_RATE,
        filter_length: int = ECHO_FILTER_LENGTH,
        ref_delay_ms: float = ECHO_REF_DELAY_MS,
        step: float = ECHO_NLMS_STEP,
    ):
        self.sample_rate = sample_rate
        self.filter_length = filter_length
        self.ref_delay = int(sample_rate * ref_delay_ms / 1000.0)
        self.step = step
        self.eps = 1e-8

        buf = max(filter_length + self.ref_delay + sample_rate * 2, 8192)
        self._ref_buf = np.zeros(buf, dtype=np.float64)
        self._ref_len = buf
        self._ref_count = 0
        self._mic_count = 0
        self._w = np.zeros(filter_length, dtype=np.float64)

    @property
    def enabled(self) -> bool:
        return ECHO_CANCELLATION_ENABLED

    def reset(self) -> None:
        self._ref_buf.fill(0.0)
        self._ref_count = 0
        self._mic_count = 0
        self._w.fill(0.0)

    def add_reference_pcm16_24k(self, pcm: bytes) -> None:
        """스피커로 출력하는 24kHz PCM 을 16kHz 참조로 적재."""
        if not pcm or librosa is None:
            return
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64) / 32768.0
        if x.size == 0:
            return
        ref = librosa.resample(
            x.astype(np.float32),
            orig_sr=RECEIVE_SAMPLE_RATE,
            target_sr=self.sample_rate,
        ).astype(np.float64)
        for sample in ref:
            self._ref_buf[self._ref_count % self._ref_len] = sample
            self._ref_count += 1

    def process_pcm16(self, pcm: bytes) -> bytes:
        """16kHz mic PCM → 에코 제거 후 int16 bytes."""
        if not pcm:
            return pcm
        mic = np.frombuffer(pcm, dtype=np.int16).astype(np.float64) / 32768.0
        out = np.empty(mic.shape[0], dtype=np.float64)
        for i, sample in enumerate(mic):
            out[i] = self._process_sample(sample)
        out_i16 = np.clip(out * 32768.0, -32768, 32767).astype(np.int16)
        return out_i16.tobytes()

    def _ref_at(self, index: int) -> float:
        if index < 0:
            return 0.0
        return float(self._ref_buf[index % self._ref_len])

    def _ref_vector(self, end_index: int) -> np.ndarray:
        start = end_index - self.filter_length + 1
        return np.array(
            [self._ref_at(i) for i in range(start, end_index + 1)],
            dtype=np.float64,
        )

    def _process_sample(self, mic_sample: float) -> float:
        ref_index = self._mic_count - self.ref_delay
        self._mic_count += 1

        if ref_index < self.filter_length:
            return mic_sample

        x = self._ref_vector(ref_index)
        echo = float(np.dot(self._w, x))
        error = mic_sample - echo
        norm = float(np.dot(x, x)) + self.eps
        self._w += (self.step * error / norm) * x
        return error


class _NoOpEchoCanceller:
    enabled = False

    def reset(self) -> None:
        return

    def add_reference_pcm16_24k(self, pcm: bytes) -> None:
        return

    def process_pcm16(self, pcm: bytes) -> bytes:
        return pcm


_echo: EchoCanceller | _NoOpEchoCanceller | None = None


def get_echo_canceller() -> EchoCanceller | _NoOpEchoCanceller:
    global _echo
    if not ECHO_CANCELLATION_ENABLED:
        return _NoOpEchoCanceller()
    if _echo is None:
        _echo = EchoCanceller()
    return _echo
