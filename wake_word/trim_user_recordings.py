#!/usr/bin/env python3
"""기존 user/*.wav 앞뒤 무음을 제거하고 2초 클립으로 다시 저장한다.

Enter 후 말하기 전 텀이 포함된 녹음은 wake word 모델이
「조용함 + 짧은 소리」 패턴을 positive 로 학습해 주변 잡음에 오탐할 수 있다.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import librosa
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
USER_DIR = BASE_DIR / "data" / "positive" / "user"
EMBEDDING_CACHE = BASE_DIR / "model" / "user_embedding.npz"

sys.path.insert(0, str(BASE_DIR.parent))
from audio.speech_trim import leading_silence_ms, trim_and_pad_int16  # noqa: E402

SAMPLE_RATE = 16000


def _save_int16(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.astype(np.int16).tobytes())


def main() -> int:
    paths = sorted(USER_DIR.glob("*.wav"))
    if not paths:
        print(f"wav 없음: {USER_DIR}")
        return 1

    print(f"처리 대상: {len(paths)}개 ({USER_DIR})")
    for path in paths:
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        before_ms = leading_silence_ms(audio, SAMPLE_RATE)
        pcm = (audio * 32767).astype(np.int16)
        trimmed = trim_and_pad_int16(pcm)
        after_ms = leading_silence_ms(trimmed, SAMPLE_RATE)
        _save_int16(path, trimmed)
        print(f"  {path.name}: lead {before_ms:.0f}ms → {after_ms:.0f}ms")

    if EMBEDDING_CACHE.is_file():
        EMBEDDING_CACHE.unlink()
        print(f"\n삭제: {EMBEDDING_CACHE.name} (다음 실행 시 임베딩 재생성)")

    print("\n완료 → python wake_word/train.py 로 wake word 모델 재학습하세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
