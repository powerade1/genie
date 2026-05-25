import math
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import wave

from config import (
    ACTIVATION_CHIME_DURATION_SEC,
    ACTIVATION_CHIME_ENABLED,
    ACTIVATION_CHIME_FREQ_HZ,
)

_chime_lock = threading.Lock()
_chime_wav_path: str | None = None
_sample_rate = 44100


def _generate_chime_wav(
    path: str,
    freq_hz: float,
    duration_sec: float,
    volume: float = 0.3,
) -> None:
    n_frames = int(_sample_rate * duration_sec)
    with wave.open(path, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(_sample_rate)
        for i in range(n_frames):
            t = i / _sample_rate
            envelope = math.exp(-t * 20)
            sample = int(volume * 32767 * envelope * math.sin(2 * math.pi * freq_hz * t))
            sample = max(-32767, min(32767, sample))
            wav_file.writeframes(struct.pack("<h", sample))


def _ensure_chime_wav() -> str | None:
    global _chime_wav_path

    with _chime_lock:
        if _chime_wav_path and os.path.isfile(_chime_wav_path):
            return _chime_wav_path

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="genie_chime_")
        os.close(fd)
        try:
            _generate_chime_wav(
                path,
                freq_hz=ACTIVATION_CHIME_FREQ_HZ,
                duration_sec=ACTIVATION_CHIME_DURATION_SEC,
            )
            _chime_wav_path = path
            return path
        except Exception as error:
            print(f"[chime] wav 생성 실패: {error}")
            try:
                os.remove(path)
            except OSError:
                pass
            return None


def _cleanup_after_play(path: str, process: subprocess.Popen) -> None:
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    except Exception:
        pass


def play_activation_chime() -> None:
    """활성화 '띠링' 효과음을 비동기 재생한다. 실패해도 앱 흐름은 중단하지 않는다."""
    if not ACTIVATION_CHIME_ENABLED:
        return

    if shutil.which("afplay") is None:
        print("[chime] afplay를 찾을 수 없습니다 (macOS 전용).")
        return

    wav_path = _ensure_chime_wav()
    if not wav_path:
        return

    try:
        process = subprocess.Popen(
            ["afplay", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as error:
        print(f"[chime] 재생 실패: {error}")
        return

    threading.Thread(
        target=_cleanup_after_play,
        args=(wav_path, process),
        daemon=True,
    ).start()
