"""본인 목소리 화자 검증 (Resemblyzer).

wake word 2차 검증 및 Live 세션 업로드 게이트에 사용한다.
등록(enrollment) 데이터: ``wake_word/data/positive/user/*.wav``
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import librosa
import numpy as np

from config import (
    LIVE_MIN_RMS,
    LIVE_SPEAKER_CHECK_INTERVAL_SEC,
    LIVE_SPEAKER_GATE_ENABLED,
    LIVE_SPEAKER_GATE_HOLD_SEC,
    SEND_SAMPLE_RATE,
    SPEAKER_EMBEDDING_PATH,
    SPEAKER_ENROLLMENT_DIR,
    SPEAKER_SIMILARITY_THRESHOLD,
    SPEAKER_VERIFICATION_ENABLED,
    SPEAKER_VERIFY_MIN_SAMPLES,
)

from audio.speech_trim import trim_silence

_ENCODER = None
_ENCODER_LOCK = threading.Lock()

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}


def rms_int16(pcm: bytes | np.ndarray) -> float:
    if isinstance(pcm, (bytes, bytearray)):
        arr = np.frombuffer(pcm, dtype=np.int16)
    else:
        arr = pcm
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))


def _get_encoder():
    global _ENCODER
    with _ENCODER_LOCK:
        if _ENCODER is None:
            from resemblyzer import VoiceEncoder

            _ENCODER = VoiceEncoder()
        return _ENCODER


def _to_float32(audio: np.ndarray, sample_rate: int, target_sr: int = 16000) -> np.ndarray | None:
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if sample_rate != target_sr:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)

    audio = trim_silence(audio)

    if len(audio) < SPEAKER_VERIFY_MIN_SAMPLES:
        return None
    return audio


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


class SpeakerVerifier:
    def __init__(
        self,
        enrollment_dir: str | Path = SPEAKER_ENROLLMENT_DIR,
        embedding_path: str | Path = SPEAKER_EMBEDDING_PATH,
        threshold: float = SPEAKER_SIMILARITY_THRESHOLD,
    ):
        self.enrollment_dir = Path(enrollment_dir)
        self.embedding_path = Path(embedding_path)
        self.threshold = threshold
        self._embedding: np.ndarray | None = None
        self._enrollment_count = 0

    @property
    def is_ready(self) -> bool:
        return self._embedding is not None

    def load(self) -> bool:
        if not SPEAKER_VERIFICATION_ENABLED:
            return False

        paths = self._collect_enrollment_paths()
        if not paths:
            print(
                "[speaker] 등록 wav 없음 — 화자 검증 비활성 "
                f"({self.enrollment_dir})"
            )
            return False

        newest_enroll = max(path.stat().st_mtime for path in paths)
        if self._load_cache(newest_enroll):
            print(
                f"[speaker] 캐시 로드 ({self._enrollment_count}개 wav, "
                f"threshold={self.threshold:.2f})"
            )
            return True

        print(f"[speaker] 임베딩 생성 중 ({len(paths)}개 wav)...")
        self._embedding = self._build_embedding(paths)
        self._enrollment_count = len(paths)
        if self._embedding is None:
            print("[speaker] 임베딩 생성 실패 — 화자 검증 비활성")
            return False

        self._save_cache(newest_enroll)
        print(
            f"[speaker] 등록 완료 ({self._enrollment_count}개 wav, "
            f"threshold={self.threshold:.2f})"
        )
        return True

    def similarity(
        self,
        audio: np.ndarray,
        sample_rate: int = SEND_SAMPLE_RATE,
    ) -> float:
        if not self.is_ready:
            return 1.0

        prepared = _to_float32(audio, sample_rate)
        if prepared is None:
            return 0.0

        encoder = _get_encoder()
        embedding = encoder.embed_utterance(prepared)
        return _cosine_similarity(embedding, self._embedding)

    def is_user(
        self,
        audio: np.ndarray,
        sample_rate: int = SEND_SAMPLE_RATE,
    ) -> tuple[bool, float]:
        if not SPEAKER_VERIFICATION_ENABLED or not self.is_ready:
            return True, 1.0
        score = self.similarity(audio, sample_rate)
        return score >= self.threshold, score

    def _collect_enrollment_paths(self) -> list[Path]:
        if not self.enrollment_dir.is_dir():
            return []
        return sorted(
            path
            for path in self.enrollment_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _AUDIO_EXTENSIONS
        )

    def _build_embedding(self, paths: list[Path]) -> np.ndarray | None:
        encoder = _get_encoder()
        embeddings: list[np.ndarray] = []
        for path in paths:
            try:
                audio, sr = librosa.load(str(path), sr=16000, mono=True)
            except Exception:
                continue
            audio = trim_silence(audio)
            if len(audio) < SPEAKER_VERIFY_MIN_SAMPLES:
                continue
            embeddings.append(encoder.embed_utterance(audio.astype(np.float32)))

        if not embeddings:
            return None

        centroid = np.mean(np.stack(embeddings, axis=0), axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        return centroid.astype(np.float32)

    def _load_cache(self, newest_enroll_mtime: float) -> bool:
        if not self.embedding_path.is_file():
            return False
        try:
            data = np.load(self.embedding_path, allow_pickle=False)
            cached_mtime = float(data["enrollment_mtime"])
            if cached_mtime < newest_enroll_mtime:
                return False
            self._embedding = np.array(data["embedding"], dtype=np.float32)
            self._enrollment_count = int(data["enrollment_count"])
            self.threshold = float(data.get("threshold", self.threshold))
            return True
        except Exception:
            return False

    def _save_cache(self, newest_enroll_mtime: float) -> None:
        if self._embedding is None:
            return
        self.embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.embedding_path,
            embedding=self._embedding,
            enrollment_mtime=newest_enroll_mtime,
            enrollment_count=self._enrollment_count,
            threshold=self.threshold,
        )


class LiveSpeakerGate:
    """Live 세션: 본인 목소리가 아니거나 너무 작은 소리는 서버에 보내지 않는다."""

    def __init__(self, verifier: SpeakerVerifier, chunk_size: int):
        self.verifier = verifier
        self.chunk_size = chunk_size
        max_chunks = max(4, (SEND_SAMPLE_RATE * 2) // chunk_size)
        self._ring: deque[np.ndarray] = deque(maxlen=max_chunks)
        self._gate_open_until = 0.0
        self._last_check = 0.0
        self._fail_streak = 0

    @property
    def enabled(self) -> bool:
        return LIVE_SPEAKER_GATE_ENABLED and self.verifier.is_ready

    def filter_chunk(self, pcm: bytes) -> bytes:
        if not self.enabled:
            return pcm

        arr = np.frombuffer(pcm, dtype=np.int16)
        self._ring.append(arr)
        now = time.monotonic()

        if now < self._gate_open_until:
            return pcm

        level = rms_int16(arr)
        if level < LIVE_MIN_RMS:
            self._fail_streak = 0
            return b"\x00" * len(pcm)

        if len(self._ring) < 4:
            return b"\x00" * len(pcm)

        if now - self._last_check < LIVE_SPEAKER_CHECK_INTERVAL_SEC:
            return b"\x00" * len(pcm)

        self._last_check = now
        audio = np.concatenate(list(self._ring))
        ok, score = self.verifier.is_user(audio, SEND_SAMPLE_RATE)
        if ok:
            self._gate_open_until = now + LIVE_SPEAKER_GATE_HOLD_SEC
            self._fail_streak = 0
            return pcm

        self._fail_streak += 1
        if self._fail_streak == 1:
            print(f"[speaker-gate] 화자 불일치 (similarity={score:.2f}), 업로드 차단")
        return b"\x00" * len(pcm)

    def force_open(self, duration_sec: float) -> None:
        """wake word 등으로 본인 확인 직후 세션 시작 시 초기 발화 지연 방지."""
        self._gate_open_until = time.monotonic() + duration_sec
        self._fail_streak = 0


_verifier: SpeakerVerifier | None = None
_verifier_lock = threading.Lock()


def get_speaker_verifier() -> SpeakerVerifier:
    global _verifier
    with _verifier_lock:
        if _verifier is None:
            _verifier = SpeakerVerifier()
            _verifier.load()
        return _verifier
