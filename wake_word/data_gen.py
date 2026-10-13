import random
import shutil
import subprocess
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

VOICES = [
    "Yuna",
    "Yuna (Enhanced)",
    "Yuna (Premium)",
    "Jian (Premium)",
    "Minsu",
    "Minsu (Enhanced)",
    "Suhyun",
    "Sora",
]
KEYWORDS = ["지니", "지니야"]
SAMPLES_PER_COMBO = 50

SAMPLE_RATE = 16000
CLIP_LENGTH = SAMPLE_RATE * 2
NOISE_MIX_PROBABILITY = 0.5
GAIN_DB_RANGE = (-20.0, -3.0)
SNR_DB_RANGE = (5.0, 20.0)
FALLBACK_NEGATIVE_LIMIT = 200

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "positive"
NOISE_DIR = BASE_DIR / "data" / "noise"
NEGATIVE_FALLBACK_DIR = BASE_DIR / "data" / "negative"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


def _db_to_linear(db: float) -> float:
    return 10 ** (db / 20.0)


def _load_noise_clips() -> list[np.ndarray]:
    sources: list[Path] = []
    if NOISE_DIR.exists():
        sources.extend(
            p for p in NOISE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        )
    if not sources and NEGATIVE_FALLBACK_DIR.exists():
        candidates = [
            p for p in NEGATIVE_FALLBACK_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        ]
        random.shuffle(candidates)
        sources.extend(candidates[:FALLBACK_NEGATIVE_LIMIT])

    clips: list[np.ndarray] = []
    for path in sources:
        try:
            audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        except Exception:
            continue
        if len(audio) < SAMPLE_RATE // 2:
            continue
        if len(audio) > CLIP_LENGTH:
            offset = random.randint(0, len(audio) - CLIP_LENGTH)
            audio = audio[offset : offset + CLIP_LENGTH]
        clips.append(audio.astype(np.float32))
    return clips


def _mix_noise(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(speech):
        repeats = len(speech) // len(noise) + 1
        noise = np.tile(noise, repeats)[: len(speech)]
    else:
        offset = random.randint(0, len(noise) - len(speech))
        noise = noise[offset : offset + len(speech)]

    sp_power = np.mean(speech**2) + 1e-9
    no_power = np.mean(noise**2) + 1e-9
    scale = np.sqrt(sp_power / (no_power * _db_to_linear(snr_db) ** 2))
    return np.clip(speech + noise * scale, -1.0, 1.0)


def _augment(audio: np.ndarray, noise_clips: list[np.ndarray]) -> np.ndarray:
    gain_db = random.uniform(*GAIN_DB_RANGE)
    augmented = np.clip(audio * _db_to_linear(gain_db), -1.0, 1.0)
    if noise_clips and random.random() < NOISE_MIX_PROBABILITY:
        noise = random.choice(noise_clips)
        snr_db = random.uniform(*SNR_DB_RANGE)
        augmented = _mix_noise(augmented, noise, snr_db)
    return augmented


def _wipe_positive_dir():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_samples():
    _wipe_positive_dir()
    noise_clips = _load_noise_clips()
    print(f"노이즈 클립 로드: {len(noise_clips)}개 (0이면 mix 스킵, 볼륨 변조만 적용)")

    idx = 0
    for keyword in KEYWORDS:
        for voice in VOICES:
            for i in range(SAMPLES_PER_COMBO):
                rate = random.randint(160, 220)
                raw_path = OUTPUT_DIR / f"_raw_{keyword}_{voice}_{i:03d}.wav"
                result = subprocess.run(
                    [
                        "say",
                        "-v",
                        voice,
                        "-r",
                        str(rate),
                        "-o",
                        str(raw_path),
                        "--data-format=LEI16@16000",
                        keyword,
                    ],
                    capture_output=True,
                )
                if result.returncode != 0 or not raw_path.exists():
                    print(f"생성 실패: {keyword} / {voice} / {i:03d}")
                    continue

                audio, _ = librosa.load(str(raw_path), sr=SAMPLE_RATE, mono=True)
                raw_path.unlink()

                augmented = _augment(audio, noise_clips)
                out_path = OUTPUT_DIR / f"{keyword}_{voice}_{i:03d}.wav"
                sf.write(str(out_path), augmented, SAMPLE_RATE, subtype="PCM_16")

                time.sleep(0.05)
                idx += 1
                if idx % 20 == 0:
                    print(f"생성 진행: {idx}개")

    print(f"\n총 {idx}개 positive sample (augmented) 생성 완료")


if __name__ == "__main__":
    generate_samples()
