import subprocess
from urllib.parse import quote


def open_url(query: str) -> str:
    if query.startswith("http"):
        url = query
    else:
        url = f"https://www.google.com/search?q={quote(query)}"

    script = (
        'tell application "Google Chrome"\n'
        '    activate\n'
        f'    open location "{url}"\n'
        'end tell'
    )
    subprocess.run(["osascript", "-e", script], check=False)
    return f"'{query}' Chrome으로 열었습니다."
