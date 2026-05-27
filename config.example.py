WAKE_WORD_MODEL_PATH = "wake_word/model/genie.onnx"
WAKE_WORD_THRESHOLD = 0.5

GEMINI_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
GEMINI_VOICE = "Aoede"

TAVILY_API_KEY = ""  # https://tavily.com 에서 발급

VAD_SILENCE_DURATION_MS = 2500
VAD_END_OF_SPEECH_SENSITIVITY = "END_SENSITIVITY_LOW"
VAD_START_OF_SPEECH_SENSITIVITY = "START_SENSITIVITY_LOW"

IDLE_TIMEOUT_SECONDS = 30

# 활성화 효과음 (띠링)
ACTIVATION_CHIME_ENABLED = True
ACTIVATION_CHIME_FREQ_HZ = 880
ACTIVATION_CHIME_DURATION_SEC = 0.15

# YouTube 재생 (yt-dlp + mpv, brew install yt-dlp mpv)
YOUTUBE_SEARCH_LIMIT = 5
YOUTUBE_AUDIO_ONLY = False
# 음성 인식 오류 보정: Gemini가 잘못 전달한 검색어 → 실제 YouTube 검색어
YOUTUBE_QUERY_ALIASES = {
    "j-pop pop": "종팝 플레이리스트",
    "j pop pop": "종팝 플레이리스트",
    "pop j-pop": "종팝 플레이리스트",
    "j-pop playlist": "종팝 플레이리스트",
    "j pop playlist": "종팝 플레이리스트",
    "j-pop": "종팝",
    "j pop": "종팝",
    "jpop": "종팝",
}

SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
WAKE_WORD_FRAME_SIZE = 1280

SYSTEM_PROMPT = (
    "당신은 맥북을 제어하는 AI 비서 '지니'입니다. "
    "사용자의 말을 듣고 시스템 제어가 필요하면 해당 함수를 호출하고, "
    "일반 질문이면 자연스러운 한국어로 답변하세요. "
    "응답은 간결하고 친근하게 합니다. "
    "한 번의 질문에는 반드시 한 번만 답하세요. 같은 내용을 반복하거나 "
    "다른 표현으로 다시 말하지 마세요. "
    "함수 호출이 필요할 때는 호출 전에 말하지 말고 함수만 호출하세요. "
    "함수 실행 결과를 받은 뒤 한 문장으로만 결과를 알려주세요. "
    "open_app, close_app 호출 시 app_name은 반드시 macOS 영어 앱 이름을 사용하세요 "
    "(예: KakaoTalk, Google Chrome, Notion). 한국어 앱 이름은 사용하지 마세요. "
    "세션 시작 직후 사용자가 먼저 말하기 전까지는 조용히 듣기만 하세요. "
    "'무엇을 도와드릴까요' 같은 선제 인사나 재질문을 하지 마세요. "
    "침묵만 있을 때는 절대 다시 말하지 마세요. "
    "모델이 학습한 시점 이후의 최신 정보, 오늘 날씨/뉴스/시세 같은 시의성 있는 "
    "질문에서는 search_web을 호출한 뒤 결과를 자연어로 요약해 답변하세요. "
    "YouTube 영상·음악·플레이리스트 재생 요청(틀어줘, 플리, 노래, 유튜브)에는 "
    "play_youtube를 사용하세요. open_url로 YouTube를 열지 마세요. "
    "play_youtube의 query는 사용자가 말한 검색어를 한국어 원문 그대로 넣으세요. "
    "발음이 비슷한 영어로 바꾸지 마세요 (예: '종팝'을 'J-POP'이나 'POP'으로 번역 금지). "
    "재생 중지 요청에는 stop_youtube를 사용하세요. "
    "코딩·개발·기능 구현·리팩터링·버그 수정 등 Supervisor와 Cursor로 진행할 "
    "작업 요청에는 create_tool을 호출하세요. tool_description에는 사용자 요청을 "
    "한국어로 명확히 요약하세요. Supervisor 서버 미실행 시 반환된 오류 메시지를 "
    "그대로 음성으로 안내하세요. "
    "사용자가 대화 종료 의사를 표하면(예: '끝', '그만', '고마워', '잘 가', '됐어') "
    "짧게 인사한 뒤 즉시 end_session 함수를 호출하세요."
)

# Supervisor 멀티에이전트 (python supervisor/main.py)
SUPERVISOR_MODEL = "gemini-3.5-flash"
SUPERVISOR_HOST = "127.0.0.1"
SUPERVISOR_PORT = 8765
CURSOR_APP_NAME = "Cursor"
CURSOR_MODE_CYCLE = ["agent", "plan", "debug", "multitask", "ask"]
CURSOR_MODE_MATCH_MAX_STEPS = 5
CURSOR_MODE_READ_DELAY_SEC = 0.35
# False면 승인 시 Shift+Tab 모드 전환을 하지 않음 (불안정할 때 권장)
CURSOR_MATCH_MODE_ON_APPROVE = False
# Cursor CLI (supervisor/cursor_cli.py)
CURSOR_CLI_PATH = "/Applications/Cursor.app/Contents/Resources/app/bin/cursor"
CURSOR_WORKSPACE = ""  # 비어 있으면 프로젝트 루트 자동
CURSOR_CLI_TRUST = True
CURSOR_CLI_PRINT = False  # True면 --print 헤드리스
SUPERVISOR_CONTEXT_PATH = "SUPERVISOR_CONTEXT.md"
SUPERVISOR_SYSTEM_PROMPT = (
    "너는 지니 Supervisor, 우진의 맥북을 제어하는 AI 비서야. "
    "말은 짧고 직설적으로 해. 불필요한 서론, 감탄사, 과장 표현 절대 금지. "
    "모르면 모른다고 해. 확실한 것만 말해. "
    "한국어로 대화하되 친근하고 편한 말투 유지. "
    "시스템 제어 요청엔 바로 실행하고 결과만 짧게 말해줘.\n\n"
    "응답은 반드시 아래 JSON 형식으로만 출력해. JSON 외 텍스트 금지.\n"
    "{\n"
    "  \"thinking\": \"요청 의도 분석 (한국어)\",\n"
    "  \"reply\": \"최종 답변 (한국어)\",\n"
    "  \"ready_for_prompt\": boolean\n"
    "}\n\n"
    "ready_for_prompt 규칙:\n"
    "- 토론·분석 중에는 false.\n"
    "- '좋아', '그렇게 하자', '진행해' 등 명확한 실행 의도 표현 시에만 true.\n"
    "- 일반 대화 중엔 프롬프트 생성 금지, ready_for_prompt만 판단."
)
SUGGEST_PROMPT_INSTRUCTION = (
    "지금까지의 대화에서 합의된 작업을 Cursor에 전달할 프롬프트로 만들어라.\n\n"
    "규칙:\n"
    "- 추가/수정할 기능을 번호 목록으로, 항목당 한 줄씩만 작성\n"
    "- 파일명·함수명은 원문 그대로, 설명은 한국어\n"
    "- 검증 방법·주의사항·배경 설명 절대 포함 금지\n"
    "- Cursor가 판단할 수 있는 건 생략\n\n"
    "출력: JSON {\"cursor_prompt\": \"...\", \"target_mode\": \"plan|agent|debug|ask\"} 형식만."
)

# Cursor 응답 후속 처리 (plan 모드 질문 자동 답변/사용자 라우팅)
CURSOR_FOLLOWUP_ENABLED = True
CURSOR_FOLLOWUP_MAX_ROUNDS = 5
CURSOR_FOLLOWUP_INSTRUCTION = (
    "다음은 Cursor Agent가 출력한 마지막 응답이다. "
    "Cursor가 사용자에게 질문하거나 결정을 요청했는지 분석하고, "
    "다음 셋 중 하나의 action으로만 응답하라.\n\n"
    "- action=\"ask_user\": 사용자의 취향/선호/외부 컨텍스트(비밀, 인프라, "
    "비즈니스 판단)가 필요한 질문이거나, 기술적으로 합리적 디폴트를 고르기 "
    "어려운 결정 요청일 때. question에 한국어로 핵심만 간결히 정리해서 채워라.\n"
    "- action=\"auto_continue\": Cursor가 질문/확인을 요청했고 사용자 의견 "
    "없이도 기술적 베스트 프랙티스로 답할 수 있을 때. auto_answer에 "
    "Cursor에 보낼 한국어 답변(명확하고 짧게)을 채워라.\n"
    "- action=\"finish\": Cursor가 더 이상 질문을 하지 않거나 작업이 끝났을 "
    "때. 후속 라운드 불필요.\n\n"
    "반드시 JSON {\"action\": ..., \"question\": ..., "
    "\"auto_answer\": ..., \"reasoning\": ...} 형식으로만 출력."
)
