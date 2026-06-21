"""LaunchAgent 자동 실행 설정 (config.py → install/wake watcher)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_launch_flags() -> tuple[bool, bool]:
    """(boot_auto_start, wake_auto_start)"""
    try:
        from config import (
            LAUNCH_AGENT_AUTO_START_ON_BOOT,
            LAUNCH_AGENT_AUTO_START_ON_WAKE,
        )

        return bool(LAUNCH_AGENT_AUTO_START_ON_BOOT), bool(
            LAUNCH_AGENT_AUTO_START_ON_WAKE
        )
    except Exception:
        return True, True


def apply_plist_run_at_load(plist_path: str | Path, enabled: bool) -> None:
    """plist 의 RunAtLoad 값을 갱신한다."""
    import plistlib

    path = Path(plist_path)
    with open(path, "rb") as f:
        data = plistlib.load(f)
    data["RunAtLoad"] = enabled
    with open(path, "wb") as f:
        plistlib.dump(data, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LaunchAgent config helper")
    parser.add_argument(
        "command",
        choices=("flags", "apply-plist"),
        help="flags: boot wake booleans / apply-plist: RunAtLoad 갱신",
    )
    parser.add_argument("plist", nargs="?", help="apply-plist 대상 plist 경로")
    args = parser.parse_args()

    boot, wake = get_launch_flags()
    if args.command == "flags":
        print(f"boot={int(boot)} wake={int(wake)}")
    else:
        if not args.plist:
            print("plist 경로 필요", file=sys.stderr)
            sys.exit(2)
        apply_plist_run_at_load(args.plist, boot)
        print(f"RunAtLoad={boot} → {args.plist}")
