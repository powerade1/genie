import time

from actions import apps, youtube

_YOUTUBE_QUERY = "종팝 플레이리스트"
_WORK_APPS = ("Cursor", "Google Chrome", "Notion")
_APP_LAUNCH_DELAY_SEC = 0.5


def start_work_routine() -> str:
    """종팝 플레이리스트 재생 후 Cursor, Chrome, Notion을 순차 실행."""
    results: list[str] = []

    results.append(youtube.play_youtube(_YOUTUBE_QUERY))

    for app_name in _WORK_APPS:
        result = apps.open_app(app_name)
        results.append(result)
        if "실패" in result:
            return f"업무 루틴 중단 ({app_name}): {result}"
        time.sleep(_APP_LAUNCH_DELAY_SEC)

    return "업무 루틴을 시작했습니다. " + " ".join(results)
