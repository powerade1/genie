#!/usr/bin/env python3
"""macOS wake 이벤트가 올 때마다 genie LaunchAgent를 깨우는 가벼운 데몬.

- `NSWorkspaceDidWakeNotification` 을 구독하고,
- 깨어나는 즉시 `launchctl kickstart -k gui/$UID/com.woojin.genie` 를 실행한다.
- 이미 실행 중이면 `-k` 옵션으로 인해 한 번 종료 후 재시작된다.
  → "잠자기 직후 종종 stale 상태였던" 케이스도 함께 청소.

사용자가 평상시(잠자기 없이) genie 를 직접 종료하면 이 watcher 는 아무 것도
하지 않으므로 의도대로 꺼진 채 유지된다.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_launch_config():
    path = Path(__file__).resolve().parent / "launch_config.py"
    spec = importlib.util.spec_from_file_location("launch_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"launch_config 로드 실패: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_launch_config = _load_launch_config()
get_launch_flags = _launch_config.get_launch_flags


GENIE_LABEL = "com.woojin.genie"
LOGVIEW_LABEL = "com.woojin.genie.logview"
# show_logs.sh 가 마지막에 `exec tail -n 80 -F …genie.out.log …` 로 자기 자신을
# 대체하므로 ps 에서는 tail 만 보인다. 이 패턴으로 logview 프로세스를 식별한다.
LOGVIEW_TAIL_NEEDLES = ("genie.out.log", "-F")
RESTART_DEBOUNCE_SEC = 2.0  # 짧은 시간 내 wake 이벤트 중복 방지


def _launchctl_kickstart(label: str) -> None:
    uid = os.getuid()
    target = f"gui/{uid}/{label}"
    cmd = ["/bin/launchctl", "kickstart", "-k", target]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print(f"[wake-watcher] kickstart OK → {target}", flush=True)
        else:
            print(
                f"[wake-watcher] kickstart 실패(code={result.returncode}): "
                f"target={target} stdout={result.stdout.strip()!r} "
                f"stderr={result.stderr.strip()!r}",
                flush=True,
            )
    except Exception as error:
        print(f"[wake-watcher] kickstart 예외 ({label}): {error}", flush=True)


def _kickstart_genie() -> None:
    _launchctl_kickstart(GENIE_LABEL)


def _scan_logview_tails() -> tuple[list[str], list[str]]:
    """logview 가 띄운 tail -F 프로세스를 ppid 기준으로 분류해 반환.

    Returns:
        (live_pids, orphan_pids) — live 는 부모(=Terminal 셸)가 살아있는 것,
        orphan 은 Terminal 창이 닫히면서 ppid=1 (launchd) 로 reparent 된 것.
    """
    try:
        result = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid=,command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return [], []

    live: list[str] = []
    orphan: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, cmd = parts
        if "tail" not in cmd:
            continue
        if not all(needle in cmd for needle in LOGVIEW_TAIL_NEEDLES):
            continue
        if ppid == "1":
            orphan.append(pid)
        else:
            live.append(pid)
    return live, orphan


def _logview_running() -> bool:
    """Terminal 창에서 보이는 logview 가 실제로 살아있으면 True."""
    live, _orphan = _scan_logview_tails()
    return bool(live)


def _cleanup_orphan_logview_tails() -> None:
    """Terminal 창은 사라지고 launchd 직속(ppid=1) 으로 남은 stale tail 정리."""
    _, orphans = _scan_logview_tails()
    for pid in orphans:
        try:
            os.kill(int(pid), 15)
            print(f"[wake-watcher] orphan logview tail({pid}) 정리", flush=True)
        except Exception:
            pass


def _ensure_logview(enabled: bool) -> None:
    """logview LaunchAgent 가 등록되어 있고 창이 살아있지 않을 때만 다시 띄운다.

    이미 창이 떠 있으면 새로 띄우지 않아 Terminal 창 누적을 방지한다.
    창이 사라졌는데 tail 만 orphan 으로 남은 케이스도 함께 정리한다.
    """
    if not enabled:
        _cleanup_orphan_logview_tails()
        return

    # logview plist 가 등록돼 있는지 확인. 사용자가 `--no-logview` 로 끈
    # 경우엔 등록 자체가 없으니 조용히 스킵.
    uid = os.getuid()
    target = f"gui/{uid}/{LOGVIEW_LABEL}"
    try:
        printed = subprocess.run(
            ["/bin/launchctl", "print", target],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if printed.returncode != 0:
            return
    except Exception:
        return

    if _logview_running():
        print("[wake-watcher] logview 이미 동작 중, 생략", flush=True)
        return

    _cleanup_orphan_logview_tails()
    _launchctl_kickstart(LOGVIEW_LABEL)


def main() -> int:
    try:
        from AppKit import NSWorkspace
        from CoreFoundation import CFRunLoopGetCurrent, CFRunLoopRun
        from Foundation import NSObject
    except Exception as error:
        print(
            f"[wake-watcher] pyobjc import 실패: {error}\n"
            "  conda activate genie 후 'pip install pyobjc-framework-Cocoa' 가 필요합니다.",
            file=sys.stderr,
        )
        return 1

    state = {"last_kick": 0.0}

    class _Handler(NSObject):
        def onWake_(self, notification):  # noqa: N802 - ObjC selector
            now = time.monotonic()
            if now - state["last_kick"] < RESTART_DEBOUNCE_SEC:
                return
            state["last_kick"] = now
            print("[wake-watcher] NSWorkspaceDidWakeNotification 수신", flush=True)
            _, wake_auto = get_launch_flags()
            if wake_auto:
                _kickstart_genie()
            else:
                print(
                    "[wake-watcher] LAUNCH_AGENT_AUTO_START_ON_WAKE=False, genie 생략",
                    flush=True,
                )
            _ensure_logview(wake_auto)

        def onSleep_(self, notification):  # noqa: N802
            print("[wake-watcher] NSWorkspaceWillSleepNotification 수신", flush=True)

    handler = _Handler.alloc().init()
    center = NSWorkspace.sharedWorkspace().notificationCenter()
    center.addObserver_selector_name_object_(
        handler, b"onWake:", "NSWorkspaceDidWakeNotification", None
    )
    center.addObserver_selector_name_object_(
        handler, b"onSleep:", "NSWorkspaceWillSleepNotification", None
    )

    boot_auto, wake_auto = get_launch_flags()
    print(
        f"[wake-watcher] 시작 (pid={os.getpid()}, uid={os.getuid()}, "
        f"target={GENIE_LABEL}, boot_auto={boot_auto}, wake_auto={wake_auto})",
        flush=True,
    )

    if boot_auto:
        _kickstart_genie()
        state["last_kick"] = time.monotonic()
    else:
        print(
            "[wake-watcher] LAUNCH_AGENT_AUTO_START_ON_BOOT=False, "
            "부팅 kickstart 생략",
            flush=True,
        )

    # boot 자동 시작이 켜진 경우에만, 부팅 직후 logview 를 한 번 보강한다.
    def _bootstrap_logview() -> None:
        if not boot_auto:
            return
        time.sleep(5.0)
        try:
            _ensure_logview(True)
        except Exception:
            traceback.print_exc()

    threading.Thread(
        target=_bootstrap_logview,
        name="logview-bootstrap",
        daemon=True,
    ).start()

    try:
        CFRunLoopGetCurrent()  # ensure run loop attached to this thread
        CFRunLoopRun()
    except KeyboardInterrupt:
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        try:
            center.removeObserver_(handler)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
