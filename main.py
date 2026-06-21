import asyncio
import logging
import os
import sys
import threading
from pathlib import Path


def _load_dotenv() -> None:
    """프로젝트 루트의 .env 파일을 os.environ 으로 로드한다.

    launchd 환경에서는 사용자 shell rc 파일이 source 되지 않아
    GEMINI_API_KEY 같은 변수가 비어 있다. 의존성 없이 표준 라이브러리만으로
    파싱한다 (단순 KEY=VALUE / 따옴표 / # 주석 / export 접두사 지원).
    """
    env_path = Path(__file__).parent / ".env"
    if not env_path.is_file():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if (len(value) >= 2) and (
                (value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")
            ):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as error:
        print(f"[env] .env 로드 실패(무시): {error}")


_load_dotenv()


from config import WAKE_WORD_MODEL_PATH  # noqa: E402  (env 로드 후 import)
from audio.feedback import play_activation_chime
from audio.speaker import get_speaker_verifier
from hotkey.listener import HotkeyListener
from live.session import ensure_api_key, run_live_session
from wake_word.detector import WakeWordDetector

_session_lock = threading.Lock()


class _SuppressTfliteWarning(logging.Filter):
    def filter(self, record):
        return "tflite runtime" not in record.getMessage()


class _PynputAccessibilityFilter(logging.Filter):
    _shown = False

    def filter(self, record):
        if "not trusted" in record.getMessage() or "accessibility" in record.getMessage().lower():
            if not _PynputAccessibilityFilter._shown:
                _PynputAccessibilityFilter._shown = True
                print(
                    "\n[단축키] Option+Space를 쓰려면 macOS에서 입력 모니터링 권한이 필요합니다.\n"
                    "  시스템 설정 → 개인정보 보호 및 보안 → 손쉬운 사용 → 입력 모니터링\n"
                    "  → 터미널(또는 Python) 허용 후 main.py를 다시 실행하세요.\n"
                )
            return False
        return True


def find_wake_word_model():
    model_path = Path(WAKE_WORD_MODEL_PATH)
    if model_path.exists():
        return str(model_path)

    tflite_path = model_path.with_suffix(".tflite")
    if tflite_path.exists():
        return str(tflite_path)

    return None


def launch_session(source: str, *, detector: WakeWordDetector):
    """Wake word 또는 단축키로 Gemini Live 세션 시작"""
    if not _session_lock.acquire(blocking=False):
        print(f"({source}) 이미 세션 진행 중, 무시")
        return
    try:
        if source == "hotkey":
            play_activation_chime()
        detector.suspend_for_session()
        print(f"지니 활성화 ({source})")
        asyncio.run(
            run_live_session(speaker_gate_warmup=(source == "wake-word"))
        )
        print("대기 중으로 복귀...")
    except KeyboardInterrupt:
        print("\n세션 중단")
    except Exception as error:
        print(f"\n세션 오류: {error}")
    finally:
        detector.resume_after_session()
        _session_lock.release()


def main():
    ensure_api_key()

    model_path = find_wake_word_model()
    if model_path is None:
        print("Wake word 모델을 찾을 수 없습니다.")
        print("1. python wake_word/data_gen.py")
        print("2. negative 샘플을 wake_word/data/negative/ 에 배치")
        print("3. python wake_word/train.py")
        sys.exit(1)

    if model_path.endswith(".onnx"):
        logging.getLogger().addFilter(_SuppressTfliteWarning())

    logging.getLogger("pynput").addFilter(_PynputAccessibilityFilter())

    print(f"Wake word 모델: {model_path}")
    get_speaker_verifier()
    detector = WakeWordDetector(model_path=model_path)
    hotkey = HotkeyListener(
        callback=lambda: launch_session("hotkey", detector=detector)
    )
    hotkey.start()
    print("단축키 활성: Option+Space")

    try:
        print("대기 중... ('지니야' 또는 Option+Space)")
        detector.listen(
            callback=lambda: launch_session("wake-word", detector=detector)
        )
    except KeyboardInterrupt:
        print("\n종료합니다.")
    finally:
        hotkey.stop()
        detector.close()


if __name__ == "__main__":
    main()
