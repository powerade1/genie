#!/usr/bin/env python3
"""본인 목소리로 wake word positive 샘플을 녹음한다.

TTS(say)로 만든 positive 샘플만으로 학습하면, 실제 목소리(특히 편안한 톤·
작은 목소리)와 어긋나 인식률이 떨어진다. 이 스크립트로 녹음한 파일은
`data/positive/user/` 에 저장되고, `train.py` 가 TTS 샘플보다 **높은
가중치**로 학습한다.

사용법:
    conda activate genie
    python wake_word/record_user.py              # 40회 녹음 (기본)
    python wake_word/record_user.py --count 60   # 60회

녹음 팁:
    - 평소 부르듯 **편안하고 자연스러운** 톤으로 "지니야" (또는 "지니")
    - 다급하게/숨 참고/속삭이듯 부르지 말 것 (오히려 negative 로 쓸 패턴)
    - 마이크 30~50cm, 조용한 환경
    - Enter 누른 **직후 바로** "지니야" (앞쪽 무음은 자동 제거됨)
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

from audio.speech_trim import trim_and_pad_int16

try:
    import pyaudio
except ImportError:
    print("pyaudio 가 필요합니다: conda activate genie")
    sys.exit(1)

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16
CLIP_SEC = 2.0
PHRASES = ("지니야", "지니")

BASE_DIR = Path(__file__).resolve().parent
USER_DIR = BASE_DIR / "data" / "positive" / "user"


def _next_index(directory: Path, phrase: str) -> int:
    prefix = f"user_{phrase}_"
    existing = [
        p.name for p in directory.glob(f"{prefix}*.wav") if p.is_file()
    ]
    indices = []
    for name in existing:
        try:
            indices.append(int(name.removeprefix(prefix).removesuffix(".wav")))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


def _save_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)


def _record_clip(pya: pyaudio.PyAudio, seconds: float) -> bytes:
    stream = pya.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=1024,
    )
    frames: list[bytes] = []
    target = int(SAMPLE_RATE / 1024 * seconds) + 1
    try:
        for _ in range(target):
            frames.append(stream.read(1024, exception_on_overflow=False))
    finally:
        stream.stop_stream()
        stream.close()
    return b"".join(frames)


def _rms_int16(pcm: bytes) -> float:
    arr = np.frombuffer(pcm, dtype=np.int16)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))


def main() -> int:
    parser = argparse.ArgumentParser(description="본인 wake word 녹음")
    parser.add_argument(
        "--count",
        type=int,
        default=40,
        help="녹음 횟수 (기본 40, phrase 2개면 번갈아)",
    )
    parser.add_argument(
        "--phrase",
        choices=("지니야", "지니", "both"),
        default="both",
        help="녹음할 호출어",
    )
    args = parser.parse_args()

    phrases = list(PHRASES) if args.phrase == "both" else [args.phrase]
    USER_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  wake word 본인 목소리 녹음")
    print(f"  저장: {USER_DIR}")
    print(f"  클립 길이: {CLIP_SEC}초 · 샘플레이트: {SAMPLE_RATE}Hz")
    print()
    print("  ★ 평소처럼 편안한 톤으로만 말하세요.")
    print("  ★ 다급/숨참/속삭임은 negative 학습용이라 positive 에 넣지 마세요.")
    print("=" * 60)
    print()

    pya = pyaudio.PyAudio()
    saved = 0
    try:
        for i in range(args.count):
            phrase = phrases[i % len(phrases)]
            idx = _next_index(USER_DIR, phrase)
            out_path = USER_DIR / f"user_{phrase}_{idx:03d}.wav"

            input(
                f"[{i + 1}/{args.count}] '{phrase}' — Enter 누르면 {CLIP_SEC}초 녹음... "
            )
            print("  녹음 중...", flush=True)
            pcm = _record_clip(pya, CLIP_SEC)
            pcm_arr = trim_and_pad_int16(pcm)
            rms = _rms_int16(pcm_arr.tobytes())
            if rms < 200:
                print(f"  ⚠ 너무 작습니다 (rms={rms:.0f}). 다시 녹음하세요.")
                continue

            _save_wav(out_path, pcm_arr.tobytes())
            saved += 1
            print(f"  ✓ 저장: {out_path.name} (rms={rms:.0f})")
            time.sleep(0.3)
    finally:
        pya.terminate()

    print()
    print(f"완료: {saved}개 저장 → python wake_word/train.py 로 재학습하세요.")
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
