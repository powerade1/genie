"""웨이크워드 오탐 분석용 이벤트 로그 (JSONL + 선택적 wav) 및 통계."""

from __future__ import annotations

import json
import threading
import wave
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from config import (
    WAKE_EVENT_LOG_DIR,
    WAKE_EVENT_LOG_ENABLED,
    WAKE_EVENT_STATS_PRINT_EVERY,
)

SAMPLE_RATE = 16000
_lock = threading.Lock()
_logger: WakeEventLogger | None = None


def _best_prediction(predictions: dict) -> tuple[str, float]:
    if not predictions:
        return "unknown", 0.0
    wake_word = max(predictions, key=predictions.get)
    return wake_word, float(predictions[wake_word])


@dataclass
class WakeStats:
    wake_detect_count: int = 0
    rejected_speaker_count: int = 0
    accepted_count: int = 0
    updated_at: str = ""

    def record(self, outcome: str) -> None:
        self.wake_detect_count += 1
        if outcome == "accepted":
            self.accepted_count += 1
        elif outcome == "rejected_speaker":
            self.rejected_speaker_count += 1
        self.updated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    @classmethod
    def from_dict(cls, data: dict) -> WakeStats:
        return cls(
            wake_detect_count=int(data.get("wake_detect_count", 0)),
            rejected_speaker_count=int(data.get("rejected_speaker_count", 0)),
            accepted_count=int(data.get("accepted_count", 0)),
            updated_at=str(data.get("updated_at", "")),
        )


class WakeEventLogger:
    def __init__(self, base_dir: str | Path = WAKE_EVENT_LOG_DIR):
        self.base_dir = Path(base_dir)
        self.captures_dir = self.base_dir / "captures"
        self.log_path = self.base_dir / "events.jsonl"
        self.stats_path = self.base_dir / "stats.json"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        self._stats = self._load_stats()

    @property
    def enabled(self) -> bool:
        return WAKE_EVENT_LOG_ENABLED

    @property
    def stats(self) -> WakeStats:
        return self._stats

    def _load_stats(self) -> WakeStats:
        if not self.stats_path.is_file():
            return WakeStats()
        try:
            data = json.loads(self.stats_path.read_text(encoding="utf-8"))
            return WakeStats.from_dict(data)
        except Exception:
            return WakeStats()

    def _save_stats(self) -> None:
        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self._stats), f, ensure_ascii=False, indent=2)

    def _maybe_print_stats(self) -> None:
        if WAKE_EVENT_STATS_PRINT_EVERY <= 0:
            return
        if self._stats.wake_detect_count % WAKE_EVENT_STATS_PRINT_EVERY != 0:
            return
        s = self._stats
        print(
            f"[wake-stats] detect={s.wake_detect_count} "
            f"accepted={s.accepted_count} rejected_speaker={s.rejected_speaker_count} "
            f"→ {self.stats_path}"
        )

    def log_detection(
        self,
        *,
        audio: np.ndarray | None,
        predictions: dict,
        outcome: str,
        speaker_similarity: float | None,
        threshold: float,
    ) -> None:
        if not self.enabled:
            return

        wake_word, wake_score = _best_prediction(predictions)
        timestamp = datetime.now().astimezone()

        wav_path: str | None = None
        if outcome == "accepted" and audio is not None and len(audio) > 0:
            stamp = (
                timestamp.strftime("%Y%m%dT%H%M%S")
                + f"{timestamp.microsecond // 1000:03d}"
            )
            wav_file = self.captures_dir / f"{stamp}_accepted_{wake_word}.wav"
            pcm = audio.astype(np.int16, copy=False)
            with wave.open(str(wav_file), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm.tobytes())
            wav_path = str(wav_file)

        record = {
            "timestamp": timestamp.isoformat(timespec="milliseconds"),
            "outcome": outcome,
            "wake_word": wake_word,
            "wake_score": round(wake_score, 4),
            "speaker_similarity": (
                round(speaker_similarity, 4) if speaker_similarity is not None else None
            ),
            "threshold": threshold,
            "wav_path": wav_path,
            "log_path": str(self.log_path),
            "stats_path": str(self.stats_path),
        }

        line = json.dumps(record, ensure_ascii=False)
        with _lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._stats.record(outcome)
            self._save_stats()

        if outcome == "accepted":
            print(
                f"[wake-event] accepted score={wake_score:.2f} "
                f"speaker={speaker_similarity if speaker_similarity is not None else 'n/a'} "
                f"→ {Path(wav_path).name if wav_path else 'no-audio'}"
            )
        else:
            print(
                f"[wake-event] rejected_speaker score={wake_score:.2f} "
                f"speaker={speaker_similarity if speaker_similarity is not None else 'n/a'} "
                f"(audio not saved)"
            )
        self._maybe_print_stats()


class _NoOpLogger:
    enabled = False

    def log_detection(self, **kwargs) -> None:
        return


def get_wake_event_logger() -> WakeEventLogger | _NoOpLogger:
    global _logger
    if not WAKE_EVENT_LOG_ENABLED:
        return _NoOpLogger()
    with _lock:
        if _logger is None:
            _logger = WakeEventLogger()
        return _logger
