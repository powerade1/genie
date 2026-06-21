"""녹음 클립 앞뒤 무음 제거 및 고정 길이 패딩."""

from __future__ import annotations

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

DEFAULT_TOP_DB = 35
DEFAULT_TARGET_SAMPLES = 16000 * 2
_FRAME = 512
_HOP = 256


def _speech_bounds(
    y: np.ndarray,
    *,
    threshold_ratio: float = 0.08,
) -> tuple[int, int]:
    """RMS 피크 대비 threshold_ratio 이상인 구간의 샘플 범위."""
    rms = librosa.feature.rms(y=y, frame_length=_FRAME, hop_length=_HOP)[0]
    if rms.size == 0:
        return 0, len(y)
    peak = float(np.max(rms))
    if peak <= 0:
        return 0, len(y)
    thr = max(peak * threshold_ratio, 1e-6)
    idx = np.where(rms > thr)[0]
    if idx.size == 0:
        return 0, len(y)
    start = max(0, int(idx[0]) * _HOP - _HOP)
    end = min(len(y), (int(idx[-1]) + 1) * _HOP + _HOP)
    return start, end


def trim_silence(
    audio: np.ndarray,
    *,
    top_db: float = DEFAULT_TOP_DB,
    threshold_ratio: float = 0.08,
) -> np.ndarray:
    """앞뒤 무음 구간을 제거한다."""
    if librosa is None or audio.size == 0:
        return audio

    if audio.dtype == np.int16:
        y = audio.astype(np.float32) / 32768.0
    else:
        y = audio.astype(np.float32)

    start, end = _speech_bounds(y, threshold_ratio=threshold_ratio)
    trimmed = y[start:end]
    if trimmed.size == 0:
        return y

    # librosa trim 으로 미세 잔여 무음 제거
    trimmed, _ = librosa.effects.trim(trimmed, top_db=top_db)
    if trimmed.size == 0:
        return y[start:end]
    return trimmed


def pad_to_length(audio: np.ndarray, target_samples: int = DEFAULT_TARGET_SAMPLES) -> np.ndarray:
    """음성을 앞에 두고 뒤를 0으로 패딩 (wake word 학습 관례)."""
    if audio.dtype == np.int16:
        work = audio
        dtype = np.int16
    else:
        work = np.clip(audio, -1.0, 1.0)
        dtype = np.float32

    if len(work) >= target_samples:
        return work[:target_samples].astype(dtype, copy=False)

    out = np.zeros(target_samples, dtype=dtype)
    out[: len(work)] = work
    return out


def trim_and_pad(
    audio: np.ndarray,
    *,
    target_samples: int = DEFAULT_TARGET_SAMPLES,
    top_db: float = DEFAULT_TOP_DB,
) -> np.ndarray:
    """무음 제거 후 고정 길이로 맞춘다."""
    trimmed = trim_silence(audio, top_db=top_db)
    return pad_to_length(trimmed, target_samples)


def trim_and_pad_int16(
    pcm: bytes | np.ndarray,
    *,
    target_samples: int = DEFAULT_TARGET_SAMPLES,
    top_db: float = DEFAULT_TOP_DB,
) -> np.ndarray:
    """int16 PCM/배열 → trim → int16 패딩."""
    if isinstance(pcm, (bytes, bytearray)):
        arr = np.frombuffer(pcm, dtype=np.int16)
    else:
        arr = pcm.astype(np.int16, copy=False)

    trimmed = trim_silence(arr, top_db=top_db)
    if trimmed.dtype != np.int16:
        trimmed = np.clip(trimmed * 32767.0, -32768, 32767).astype(np.int16)
    return pad_to_length(trimmed, target_samples).astype(np.int16)


def leading_silence_ms(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> float:
    """디버그용: 최대 RMS 대비 10% 이상인 첫 프레임까지 무음 길이(ms)."""
    if librosa is None or audio.size == 0:
        return 0.0

    if audio.dtype == np.int16:
        y = audio.astype(np.float32) / 32768.0
    else:
        y = audio.astype(np.float32)

    rms = librosa.feature.rms(y=y, frame_length=_FRAME, hop_length=_HOP)[0]
    if rms.size == 0 or np.max(rms) <= 0:
        return 0.0
    speech = np.where(rms > np.max(rms) * 0.08)[0]
    if speech.size == 0:
        return len(y) / sample_rate * 1000
    return float(speech[0] * _HOP / sample_rate * 1000)
