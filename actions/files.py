import os
import subprocess


def file_operation(action: str, path: str) -> str:
    expanded = os.path.expanduser(path)

    if action == "open":
        subprocess.run(["open", expanded], check=False)
        return f"'{path}' 열었습니다."
    if action == "reveal":
        subprocess.run(["open", "-R", expanded], check=False)
        return f"'{path}' 위치를 Finder에서 열었습니다."
    if action == "trash":
        script = f'tell application "Finder" to delete POSIX file "{expanded}"'
        subprocess.run(["osascript", "-e", script], check=False)
        return f"'{path}'를 휴지통으로 이동했습니다."
    return f"지원하지 않는 파일 작업입니다: {action}"
