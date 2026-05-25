# Supervisor 컨텍스트 (SSOT)

> Supervisor AI 세션 시작 시 이 파일 전체가 system prompt에 주입됩니다.  
> 상세 기획·구현 가이드는 [`PRD.md`](PRD.md) 참고.

## 한 줄 요약

macOS **음성 비서 Genie**와, Cursor 협업용 **Supervisor 웹 UI**(Gemini)가 공존하는 monorepo.

## Genie — `python main.py`

| 항목 | 내용 |
| --- | --- |
| 트리거 | wake word `지니`/`지니야` (`wake_word/`, ONNX 모델) · **Option+Space** (`hotkey/`) |
| AI | Gemini **Live native audio** — STT·추론·TTS·tool call 통합 (`live/session.py`) |
| 도구 | 앱 실행/종료 · 볼륨/밝기 · URL · 파일(open/reveal/trash) · 웹검색 · YouTube 재생/중지 · 세션 종료 |
| UX | wake word 즉시 / 단축키 시 activation chime (`audio/feedback.py`) · 유휴 타임아웃 |
| 설정 | `config.py` · `GEMINI_API_KEY` 환경변수 |

## Supervisor — `python supervisor/main.py` (:8765)

| 항목 | 내용 |
| --- | --- |
| 역할 | 아키텍트형 **Gemini 3.5 Flash** — 토론 → Cursor용 실행 프롬프트 생성 |
| UI | 3단: 세션(localStorage) · 채팅(thinking 접기) · Cursor 프롬프트 패널 |
| API | `POST /chat` · `/suggest-prompt` · `/reset` · `/seed` |
| 프롬프트 전달 | **클립보드 복사 버튼** (osascript 자동 주입은 폐기 — 포커스·세션·모드 불안정) |
| 컨텍스트 | 세션 시작(`reset_chat`)마다 이 파일 로드 (`supervisor/context.py`) |
| suggest JSON | `cursor_prompt` + `target_mode` (`plan` \| `agent` \| `debug` \| `ask`) |

## 주요 경로

`main.py` · `wake_word/` · `live/` · `actions/` · `hotkey/` · `audio/` · `supervisor/` · `config.py` · `MultiAgent.md`

## 제약

브라우저 DOM 제어 불가 · 밝기=`brightness` CLI · Live=온라인 필수 · wake word 오탐 가능

## 업데이트 히스토리

| 날짜 | 변경 |
| --- | --- |
| 2025-05 | Genie v1: wake word + Gemini Live + actions 레이어 |
| 2025-05 | Option+Space, activation chime, `search_web`/`play_youtube` |
| 2025-05 | Supervisor Phase 1: FastAPI 3단 UI, suggest/reset/seed, JSON thinking |
| 2025-05 | Cursor 자동 주입(osascript·모드 매칭) 시도 → **복사 버튼**으로 대체 |
| 2025-05 | 컨텍스트를 `SUPERVISOR_CONTEXT.md`로 분리 — PRD와 독립 관리 |
