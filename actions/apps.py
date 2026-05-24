import subprocess

# macOS open -a / AppleScript는 영어 앱 이름 필요
APP_NAME_ALIASES = {
    "카카오톡": "KakaoTalk",
    "카톡": "KakaoTalk",
    "크롬": "Google Chrome",
    "구글 크롬": "Google Chrome",
    "사파리": "Safari",
    "노션": "Notion",
    "슬랙": "Slack",
    "디스코드": "Discord",
    "메시지": "Messages",
    "메모": "Notes",
    "설정": "System Settings",
    "시스템 설정": "System Settings",
    "파인더": "Finder",
    "스포티파이": "Spotify",
    "유튜브": "Google Chrome",
    "텔레그램": "Telegram",
    "줌": "zoom.us",
    "포토샵": "Adobe Photoshop",
    "피그마": "Figma",
    "vscode": "Visual Studio Code",
    "코드": "Visual Studio Code",
    "터미널": "Terminal",
    "음악": "Music",
    "캘린더": "Calendar",
    "메일": "Mail",
    "xcode": "Xcode",
}


def resolve_app_name(app_name: str) -> str:
    name = app_name.strip()
    if name in APP_NAME_ALIASES:
        return APP_NAME_ALIASES[name]
    lowered = name.lower()
    for alias, resolved in APP_NAME_ALIASES.items():
        if alias.lower() == lowered:
            return resolved
    return name


def open_app(app_name: str) -> str:
    resolved = resolve_app_name(app_name)
    result = subprocess.run(
        ["open", "-a", resolved],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        return (
            f"'{resolved}' 실행 실패: {err or '앱을 찾을 수 없습니다'}. "
            "macOS 영어 앱 이름을 사용하세요 (예: KakaoTalk, Google Chrome)."
        )
    return f"{resolved} 실행했습니다."


def close_app(app_name: str) -> str:
    resolved = resolve_app_name(app_name)
    script = f'tell application "{resolved}" to quit'
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        return f"'{resolved}' 종료 실패: {err or '앱을 찾을 수 없습니다'}."
    return f"{resolved} 종료했습니다."
