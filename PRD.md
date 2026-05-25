# 🤖 Genie — 맥북 AI 음성 비서

> "지니야" 한마디로 시작하는, 말로 뭐든 되는 맥북 비서
> 

> Supervisor AI 컨텍스트(SSOT)는 [`SUPERVISOR_CONTEXT.md`](SUPERVISOR_CONTEXT.md)에서 별도 관리합니다.

---

## 📐 전체 아키텍처

```
[마이크 상시 대기]
       │
       ▼
[Wake Word 감지] ← openWakeWord 커스텀 모델 (로컬)
  "지니" / "지니야"
       │
       ▼
[Gemini Live API 연결]
  ┌─────────────────────────────────────┐
  │  오디오 입력 → Gemini 처리          │
  │  STT + 의도파악 + Function Calling  │
  │  + TTS 응답 전부 통합               │
  │  (gemini-2.5-flash-native-audio)    │
  │  무료                               │
  └─────────────────────────────────────┘
       │
       ▼
[실행 레이어]
  AppleScript / shell / Python
  앱 제어 · 볼륨 · 밝기 · 파일 · 브라우저
       │
       ▼
[오디오 응답 출력]
  Gemini가 직접 생성한 음성 재생
```

### 기존 대비 달라진 점

| 항목 | 기존 | 변경 후 |
| --- | --- | --- |
| STT | macOS 내장 | Gemini Live API 통합 |
| TTS | macOS say (Yuna) | Gemini Live API 통합 |
| 두뇌 | Gemini 2.5 Flash (텍스트) | Gemini 2.5 Flash Native Audio |
| 파이프라인 단계 | 5단계 | 3단계 |
| 비용 | 무료 | 무료 |

---

## 📁 프로젝트 구조

```
genie/
├── main.py                  # 메인 루프 (전체 파이프라인 통합)
├── wake_word/
│   ├── train.py             # 모델 학습 스크립트
│   ├── data_gen.py          # TTS 합성 데이터 생성
│   ├── model/
│   │   └── genie.tflite     # 학습된 wake word 모델
│   └── detector.py          # 실시간 감지 루프
├── live/
│   └── session.py           # Gemini Live API 세션 관리
│                            # (STT + 두뇌 + TTS 통합)
├── actions/
│   ├── apps.py              # 앱 실행/종료
│   ├── system.py            # 볼륨, 밝기, 전원
│   ├── files.py             # 파일 조작
│   └── browser.py           # 브라우저 제어
├── config.py                # API 키, 설정값
└── requirements.txt
```

---

## 🛠️ 사전 준비

### 필수 라이브러리 설치

```bash
pip install openwakeword
pip install pyaudio
pip install google-generativeai
pip install librosa soundfile
```

### API 키 설정

`config.py` 파일 생성:

```python
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

> ⚠️ `.gitignore`에 `config.py` 반드시 추가
> 

---

## 1단계: Wake Word 모델 학습

### 1-1. TTS 합성 데이터 생성

`wake_word/data_gen.py`

"지니" / "지니야" 두 패턴 모두 학습하기 위해 macOS 내장 TTS로 다양한 목소리 변형을 자동 생성한다.

```python
import subprocess
import os
import random

# macOS에서 사용 가능한 한국어/다국어 TTS 목소리 목록
# `say -v ?` 명령어로 전체 목록 확인 가능
VOICES = ["Yuna", "Kyoko", "Nora", "Tessa", "Samantha"]
KEYWORDS = ["지니", "지니야"]
OUTPUT_DIR = "wake_word/data/positive"
SAMPLES_PER_COMBO = 50  # 목소리 × 키워드 × 50개 = 충분한 데이터

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_samples():
    idx = 0
    for keyword in KEYWORDS:
        for voice in VOICES:
            for i in range(SAMPLES_PER_COMBO):
                # 속도/피치 랜덤 변형으로 다양성 확보
                rate = random.randint(160, 220)
                output_path = f"{OUTPUT_DIR}/{keyword}_{voice}_{i:03d}.wav"
                subprocess.run([
                    "say", "-v", voice,
                    "-r", str(rate),
                    "-o", output_path,
                    "--data-format=LEF32@16000",
                    keyword
                ])
                idx += 1
                print(f"생성 중: {idx}개 완료 ({output_path})")

    print(f"\n✅ 총 {idx}개 positive sample 생성 완료")

if __name__ == "__main__":
    generate_samples()
```

> Negative sample은 [Mozilla Common Voice 한국어](https://commonvoice.mozilla.org/ko)에서 다운로드하여 `wake_word/data/negative/`에 배치
> 

### 1-2. 모델 학습

`wake_word/train.py`

openWakeWord의 커스텀 학습 파이프라인을 사용한다.

```python
from openwakeword.train import train_model

train_model(
    positive_dir="wake_word/data/positive",   # "지니" + "지니야" 합성 샘플
    negative_dir="wake_word/data/negative",   # Mozilla Common Voice 한국어
    output_path="wake_word/model/genie.tflite",
    epochs=50,
    batch_size=32
)
```

```bash
python wake_word/train.py
```

> 학습 완료 후 `wake_word/model/genie.tflite` 생성 확인
> 

### 1-3. 실시간 Wake Word 감지

`wake_word/detector.py`

```python
import pyaudio
import numpy as np
from openwakeword.model import Model

class WakeWordDetector:
    def __init__(self, model_path="wake_word/model/genie.tflite", threshold=0.5):
        self.model = Model(wakeword_models=[model_path])
        self.threshold = threshold
        self.audio = pyaudio.PyAudio()

    def listen(self, callback):
        """Wake word 감지 시 callback 실행"""
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1280
        )
        print("🎤 대기 중... ('지니야'라고 불러보세요)")

        while True:
            audio_chunk = stream.read(1280, exception_on_overflow=False)
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            predictions = self.model.predict(audio_data)

            for model_name, score in predictions.items():
                if score > self.threshold:
                    print(f"✅ Wake word 감지! (score: {score:.2f})")
                    callback()
                    self.model.reset()  # 연속 감지 방지
                    break
```

---

## 2단계: Gemini Live API 세션

`live/session.py`

STT, 두뇌(Function Calling), TTS가 전부 이 하나로 통합된다. 오디오를 직접 주고받으므로 별도 STT/TTS 모듈이 필요없다.

```python
import asyncio
import pyaudio
import google.genai as genai
from google.genai.types import LiveConnectConfig, SpeechConfig, VoiceConfig
from config import GEMINI_API_KEY
from actions import apps, system, browser, files

genai.configure(api_key=GEMINI_API_KEY)

# 사용 가능한 도구(함수) 정의
TOOLS = [
    {
        "name": "open_app",
        "description": "맥북에서 애플리케이션을 실행합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {"type": "string", "description": "앱 이름 (예: Safari, Notion, Spotify)"}
            },
            "required": ["app_name"]
        }
    },
    {
        "name": "close_app",
        "description": "실행 중인 애플리케이션을 종료합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {"type": "string", "description": "종료할 앱 이름"}
            },
            "required": ["app_name"]
        }
    },
    {
        "name": "set_volume",
        "description": "시스템 볼륨을 조절합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "level": {"type": "integer", "description": "볼륨 레벨 0~100"},
                "direction": {"type": "string", "enum": ["up", "down", "mute"]}
            }
        }
    },
    {
        "name": "set_brightness",
        "description": "화면 밝기를 조절합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "level": {"type": "integer", "description": "밝기 레벨 0~100"},
                "direction": {"type": "string", "enum": ["up", "down"]}
            }
        }
    },
    {
        "name": "open_url",
        "description": "브라우저에서 URL 또는 검색어를 엽니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "URL 또는 검색어"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "file_operation",
        "description": "파일을 열거나 폴더를 탐색합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["open", "reveal", "trash"]},
                "path": {"type": "string", "description": "파일 또는 폴더 경로"}
            },
            "required": ["action", "path"]
        }
    }
]

ACTION_MAP = {
    "open_app": apps.open_app,
    "close_app": apps.close_app,
    "set_volume": system.set_volume,
    "set_brightness": system.set_brightness,
    "open_url": browser.open_url,
    "file_operation": files.file_operation,
}

SYSTEM_PROMPT = (
    "당신은 맥북을 제어하는 AI 비서 '지니'입니다. "
    "사용자의 말을 듣고 시스템 제어가 필요하면 해당 함수를 호출하고, "
    "일반 질문이면 자연스러운 한국어로 답변하세요. "
    "응답은 간결하고 친근하게 합니다."
)

async def run_live_session():
    """Wake word 감지 후 Gemini Live API 세션 실행"""
    client = genai.Client()

    config = LiveConnectConfig(
        response_modalities=["AUDIO"],   # 음성으로 응답
        speech_config=SpeechConfig(
            voice_config=VoiceConfig(prebuilt_voice_config={"voice_name": "Aoede"})
        ),
        system_instruction=SYSTEM_PROMPT,
        tools=TOOLS
    )

    audio = pyaudio.PyAudio()

    async with client.aio.live.connect(
        model="gemini-2.5-flash-native-audio-preview-12-2025",
        config=config
    ) as session:

        print("🟢 Gemini Live 연결됨. 말씀하세요.")

        # 마이크 입력 스트림
        mic_stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )

        # 스피커 출력 스트림
        speaker_stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True
        )

        async def send_audio():
            """마이크 → Gemini"""
            while True:
                chunk = mic_stream.read(1024, exception_on_overflow=False)
                await session.send({"mime_type": "audio/pcm", "data": chunk})
                await asyncio.sleep(0)

        async def receive():
            """Gemini → 스피커 + Function Call 처리"""
            async for response in session.receive():
                # 오디오 응답 재생
                if response.data:
                    speaker_stream.write(response.data)

                # Function Call 처리
                if response.tool_call:
                    for fc in response.tool_call.function_calls:
                        func_name = fc.name
                        args = dict(fc.args)
                        print(f"[실행]: {func_name}({args})")

                        if func_name in ACTION_MAP:
                            try:
                                result = ACTION_MAP[func_name](**args)
                                # 실행 결과를 Gemini에 전달
                                await session.send_tool_response(
                                    function_responses=[{
                                        "id": fc.id,
                                        "name": func_name,
                                        "response": {"result": result}
                                    }]
                                )
                            except Exception as e:
                                print(f"실행 오류: {e}")

                # 세션 종료 신호
                if response.server_content and response.server_content.turn_complete:
                    break

        await asyncio.gather(send_audio(), receive())

        mic_stream.stop_stream()
        mic_stream.close()
        speaker_stream.stop_stream()
        speaker_stream.close()
```

---

## 3단계: 실행 레이어

### 앱 제어 `actions/apps.py`

```python
import os
import subprocess

def open_app(app_name: str) -> str:
    os.system(f"open -a '{app_name}'")
    return f"{app_name} 실행했습니다."

def close_app(app_name: str) -> str:
    script = f'tell application "{app_name}" to quit'
    subprocess.run(["osascript", "-e", script])
    return f"{app_name} 종료했습니다."
```

### 시스템 제어 `actions/system.py`

```python
import subprocess

def set_volume(level: int = None, direction: str = None) -> str:
    if direction == "mute":
        subprocess.run(["osascript", "-e", "set volume output muted true"])
        return "음소거했습니다."
    elif direction == "up":
        subprocess.run(["osascript", "-e",
            "set volume output volume (output volume of (get volume settings) + 10)"])
        return "볼륨을 높였습니다."
    elif direction == "down":
        subprocess.run(["osascript", "-e",
            "set volume output volume (output volume of (get volume settings) - 10)"])
        return "볼륨을 낮췄습니다."
    elif level is not None:
        subprocess.run(["osascript", "-e", f"set volume output volume {level}"])
        return f"볼륨을 {level}으로 설정했습니다."

def set_brightness(level: int = None, direction: str = None) -> str:
    # brightness 제어는 brightness CLI 도구 필요: brew install brightness
    if direction == "up":
        subprocess.run(["brightness", "0.1", "+"])
        return "밝기를 높였습니다."
    elif direction == "down":
        subprocess.run(["brightness", "0.1", "-"])
        return "밝기를 낮췄습니다."
    elif level is not None:
        normalized = level / 100
        subprocess.run(["brightness", str(normalized)])
        return f"밝기를 {level}%로 설정했습니다."
```

### 브라우저 제어 `actions/browser.py`

```python
import subprocess

def open_url(query: str) -> str:
    # URL이면 그대로, 아니면 구글 검색
    if query.startswith("http"):
        url = query
    else:
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"

    script = f'tell application "Safari" to open location "{url}"'
    subprocess.run(["osascript", "-e", script])
    return f"'{query}' 열었습니다."
```

### 파일 조작 `actions/files.py`

```python
import os
import subprocess

def file_operation(action: str, path: str) -> str:
    expanded = os.path.expanduser(path)

    if action == "open":
        subprocess.run(["open", expanded])
        return f"'{path}' 열었습니다."
    elif action == "reveal":
        subprocess.run(["open", "-R", expanded])
        return f"'{path}' 위치를 Finder에서 열었습니다."
    elif action == "trash":
        script = f'tell application "Finder" to delete POSIX file "{expanded}"'
        subprocess.run(["osascript", "-e", script])
        return f"'{path}'를 휴지통으로 이동했습니다."
```

---

## 4단계: 메인 루프 통합

`main.py`

모든 모듈을 연결한다. wake word 감지 후 Gemini Live API 세션을 시작하는 것이 전부다.

```python
import asyncio
from wake_word.detector import WakeWordDetector
from live.session import run_live_session

detector = WakeWordDetector()

def on_wake_word():
    """Wake word 감지 시 Gemini Live 세션 실행"""
    print("✅ 지니 활성화")
    asyncio.run(run_live_session())
    print("💤 대기 중으로 복귀...")

def main():
    print("💤 대기 중... ('지니야'라고 불러보세요)")
    detector.listen(callback=on_wake_word)

if __name__ == "__main__":
    main()
```

---

## 📈 향후 확장 방향

### 단기

- **대화 맥락 유지**: `agent.py`의 chat history 활용해 멀티턴 대화
- **개인 목소리 fine-tuning**: 본인 목소리 20~30개 녹음 추가 시 정확도 향상
- **알림/리마인더**: `at` 명령어 또는 macOS 캘린더 연동

### 중기

- **Gemini 검색 연동**: `google_search` function 추가로 실시간 정보 조회
- **브라우저 깊은 제어**: AppleScript → Playwright로 교체 시 로그인/폼 입력 가능
- **다중 모델 지원**: config에서 Claude / GPT-4o로 교체 가능한 구조

### 장기

- **iOS 단축어 연동**: 아이폰에서도 같은 비서 호출
- **wake word 정확도 개선**: 본인 목소리 데이터 누적 → 주기적 재학습

---

## ⚠️ 알려진 한계

| 항목 | 내용 |
| --- | --- |
| 브라우저 제어 | 앱 실행/URL 열기까지. 페이지 내 상호작용은 Playwright 필요 |
| 밝기 제어 | `brightness` CLI 별도 설치 필요 (`brew install brightness`) |
| Gemini Live API | 인터넷 연결 필요. 오프라인 환경에서는 동작 불가 |
| Wake word 오탐 | 초기 모델은 오탐 가능. 본인 목소리 데이터 추가 시 개선 |
| 인텔 맥북 부하 | Wake word 상시 감지 + Gemini API 호출 시 팬 소음 가능 |