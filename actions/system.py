import shutil
import subprocess


def set_volume(level: int = None, direction: str = None) -> str:
    if direction == "mute":
        subprocess.run(["osascript", "-e", "set volume output muted true"], check=False)
        return "음소거했습니다."
    if direction == "up":
        subprocess.run(
            [
                "osascript",
                "-e",
                "set volume output volume (output volume of (get volume settings) + 10)",
            ],
            check=False,
        )
        return "볼륨을 높였습니다."
    if direction == "down":
        subprocess.run(
            [
                "osascript",
                "-e",
                "set volume output volume (output volume of (get volume settings) - 10)",
            ],
            check=False,
        )
        return "볼륨을 낮췄습니다."
    if level is not None:
        subprocess.run(["osascript", "-e", f"set volume output volume {level}"], check=False)
        return f"볼륨을 {level}으로 설정했습니다."
    return "볼륨 설정을 확인할 수 없습니다."


def set_brightness(level: int = None, direction: str = None) -> str:
    if not shutil.which("brightness"):
        return "밝기 제어를 위해 'brew install brightness'를 먼저 실행해주세요."

    if direction == "up":
        subprocess.run(["brightness", "0.1", "+"], check=False)
        return "밝기를 높였습니다."
    if direction == "down":
        subprocess.run(["brightness", "0.1", "-"], check=False)
        return "밝기를 낮췄습니다."
    if level is not None:
        normalized = level / 100
        subprocess.run(["brightness", str(normalized)], check=False)
        return f"밝기를 {level}%로 설정했습니다."
    return "밝기 설정을 확인할 수 없습니다."
