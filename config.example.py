WAKE_WORD_MODEL_PATH = "wake_word/model/genie.onnx"
WAKE_WORD_THRESHOLD = 0.5

GEMINI_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
GEMINI_VOICE = "Aoede"

TAVILY_API_KEY = ""  # https://tavily.com 에서 발급

VAD_SILENCE_DURATION_MS = 2500
VAD_END_OF_SPEECH_SENSITIVITY = "END_SENSITIVITY_LOW"
VAD_START_OF_SPEECH_SENSITIVITY = "START_SENSITIVITY_LOW"

IDLE_TIMEOUT_SECONDS = 30

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
    "사용자가 대화 종료 의사를 표하면(예: '끝', '그만', '고마워', '잘 가', '됐어') "
    "짧게 인사한 뒤 즉시 end_session 함수를 호출하세요."
)
