import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

try:
    from config import YOUTUBE_AUDIO_ONLY, YOUTUBE_QUERY_ALIASES, YOUTUBE_SEARCH_LIMIT
except ImportError:
    YOUTUBE_SEARCH_LIMIT = 5
    YOUTUBE_AUDIO_ONLY = False
    YOUTUBE_QUERY_ALIASES = {}

_PRINT_FMT = "%(title)s|||%(webpage_url)s|||%(duration)s"
_INSTALL_HINT = "brew install yt-dlp mpv"
_AUDIO_RELEASE_DELAY_SEC = 0.5


@dataclass
class PlayItem:
    title: str
    url: str
    duration: int | None = None


def _check_dependencies() -> str | None:
    if not shutil.which("yt-dlp"):
        return f"yt-dlp가 설치되어 있지 않습니다. {_INSTALL_HINT}"
    if not shutil.which("mpv"):
        return f"mpv가 설치되어 있지 않습니다. {_INSTALL_HINT}"
    return None


def _parse_duration(raw: str) -> int | None:
    raw = raw.strip()
    if not raw or raw.upper() == "NA":
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _canonical_query_key(query: str) -> str:
    return re.sub(r"[\s,]+", " ", query.strip().lower())


def _normalize_youtube_query(query: str) -> str:
    """Gemini 음성 오인식 보정 (config.YOUTUBE_QUERY_ALIASES)."""
    raw = query.strip()
    if not raw:
        return raw

    key = _canonical_query_key(raw)
    aliases = {_canonical_query_key(k): v for k, v in YOUTUBE_QUERY_ALIASES.items()}
    if key in aliases:
        return aliases[key]

    if re.fullmatch(r"j[\s-]?pop(?:\s+pop)?(?:\s+playlist)?", key):
        parts = key.split()
        if len(parts) >= 3 or "playlist" in parts:
            return "종팝 플레이리스트"
        return "종팝"
    if re.fullmatch(r"pop\s+j[\s-]?pop(?:\s+playlist)?", key):
        return "종팝 플레이리스트"

    return raw


class YouTubePlayer:
    def __init__(self):
        self._lock = threading.Lock()
        self._playback_thread: threading.Thread | None = None
        self._playback_stop = threading.Event()
        self._queue: list[PlayItem] = []
        self._queue_index = 0
        self._current: PlayItem | None = None
        self._process: subprocess.Popen | None = None
        self._ipc_path: str | None = None

    def _cleanup_ipc(self) -> None:
        if not self._ipc_path:
            return
        try:
            os.unlink(self._ipc_path)
        except FileNotFoundError:
            pass

    def _new_ipc_path(self) -> str:
        return f"/tmp/genie-youtube-{os.getpid()}-{time.time_ns()}.ipc"
    def search(self, query: str, limit: int | None = None) -> list[PlayItem]:
        limit = limit or YOUTUBE_SEARCH_LIMIT
        result = subprocess.run(
            [
                "yt-dlp",
                "--flat-playlist",
                "--print",
                _PRINT_FMT,
                f"ytsearch{limit}:{query}",
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(err or "YouTube 검색 실패")

        items: list[PlayItem] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|||")
            if len(parts) < 2:
                continue
            title = parts[0].strip()
            url = parts[1].strip()
            duration = _parse_duration(parts[2]) if len(parts) > 2 else None
            if url:
                items.append(PlayItem(title=title or url, url=url, duration=duration))
        return items

    def _resolve_playback_target(self, page_url: str) -> tuple[str, bool]:
        """(재생 URL, mpv --ytdl 사용 여부) 반환."""
        if YOUTUBE_AUDIO_ONLY:
            if "googlevideo.com" in page_url:
                return page_url, False
            if "playlist?list=" in page_url and "watch?" not in page_url:
                return page_url, True
            result = subprocess.run(
                ["yt-dlp", "-f", "bestaudio/best", "-g", "--no-playlist", page_url],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                if lines:
                    return lines[0], False
            return page_url, True

        # 영상: DASH는 video/audio URL이 분리됨 → 페이지 URL + ytdl로 mux
        return page_url, True

    def _mpv_ipc_send(self, command: list) -> bool:
        if not self._ipc_path or not os.path.exists(self._ipc_path):
            return False
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect(self._ipc_path)
            sock.sendall((json.dumps({"command": command}) + "\n").encode())
            sock.close()
            return True
        except OSError:
            return False

    def _kill_mpv(self):
        self._mpv_ipc_send(["quit"])
        proc = self._process
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        self._process = None
        self._cleanup_ipc()
        self._ipc_path = None
        time.sleep(_AUDIO_RELEASE_DELAY_SEC)

    def _build_mpv_command(self, stream_url: str, *, use_ytdl: bool, ipc_path: str) -> list[str]:
        mpv_bin = shutil.which("mpv") or "mpv"
        cmd = [
            mpv_bin,
            "--input-default-bindings",  # 스페이스바 = 정지/재생 등 기본 단축키
            "--msg-level=all=no",  # mpv 로그 스팸 차단
            "--no-terminal",
            "--really-quiet",
            "--no-config",
            "--keep-open=no",
            "--idle=no",
            f"--input-ipc-server={ipc_path}",
            "--title=genie",
        ]
        if use_ytdl:
            cmd.append("--ytdl=yes")
        else:
            cmd.append("--ytdl=no")

        if YOUTUBE_AUDIO_ONLY:
            cmd.extend(
                [
                    "--no-video",
                    "--vo=null",
                    "--force-window=no",
                    "--audio-display=no",
                    "--macos-app-activation-policy=accessory",
                    # 오디오 전용: UI 불필요
                    "--osd-level=0",
                    "--osc=no",
                    "--load-scripts=no",
                ]
            )
        elif sys.platform == "darwin":
            cmd.extend(
                [
                    "--vo=gpu",
                    "--force-window=yes",
                    # 영상: OSC(재생바·버튼) + OSD 표시
                    "--osc=yes",
                    "--osd-level=1",
                    "--osd-font=Helvetica",
                ]
            )
        else:
            cmd.extend(
                [
                    "--force-window=yes",
                    "--osc=yes",
                    "--osd-level=1",
                ]
            )

        cmd.append(stream_url)
        return cmd

    def _launch_mpv(self, stream_url: str, *, use_ytdl: bool) -> None:
        self._cleanup_ipc()
        self._ipc_path = self._new_ipc_path()

        cmd = self._build_mpv_command(
            stream_url, use_ytdl=use_ytdl, ipc_path=self._ipc_path
        )
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            env={**os.environ, "OS_ACTIVITY_MODE": "disable"},
        )

    def _wait_mpv_exit(self) -> None:
        proc = self._process
        if proc is None:
            return
        while not self._playback_stop.is_set():
            try:
                proc.wait(timeout=0.3)
                self._process = None
                self._cleanup_ipc()
                self._ipc_path = None
                return
            except subprocess.TimeoutExpired:
                continue
        # stop_youtube 등으로 중단된 경우 _kill_mpv()가 프로세스 정리

    def _playback_loop(self):
        while not self._playback_stop.is_set():
            with self._lock:
                if self._queue_index >= len(self._queue):
                    self._current = None
                    return
                item = self._queue[self._queue_index]
                self._queue_index += 1
                self._current = item

            playback_url, use_ytdl = self._resolve_playback_target(item.url)
            queue_pos = self._queue_index
            queue_total = len(self._queue)
            print(
                f"[youtube] 재생 {queue_pos}/{queue_total}: {item.title}",
                file=sys.stderr,
            )
            try:
                self._launch_mpv(playback_url, use_ytdl=use_ytdl)
            except OSError as error:
                print(f"[youtube] mpv 실행 실패: {error}", file=sys.stderr)
                continue

            self._wait_mpv_exit()

            if self._playback_stop.is_set():
                return

            time.sleep(_AUDIO_RELEASE_DELAY_SEC)

    def _start_playback(self, items: list[PlayItem]) -> PlayItem | None:
        if not items:
            return None

        self._playback_stop.clear()
        self._queue = items
        self._queue_index = 0
        self._playback_thread = threading.Thread(
            target=self._playback_loop, daemon=True, name="youtube-playback"
        )
        self._playback_thread.start()
        return items[0]

    def _stop_playback(self):
        self._playback_stop.set()
        self._kill_mpv()
        self._queue = []
        self._queue_index = 0
        self._current = None
        thread = self._playback_thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    def play_query(self, query: str, *, replace: bool = True) -> str:
        missing = _check_dependencies()
        if missing:
            return missing

        try:
            items = self.search(query)
        except Exception as error:
            return f"검색 실패: {error}"

        if not items:
            return f"'{query}'에 대한 재생할 영상을 찾지 못했습니다."

        with self._lock:
            if replace:
                self._stop_playback()
            started = self._start_playback(items)

        if started is None:
            return "재생을 시작하지 못했습니다."

        mode = "소리만" if YOUTUBE_AUDIO_ONLY else "영상"
        queued = len(items) - 1
        suffix = f" (대기 {queued}개)" if queued > 0 else ""
        return f"재생 중 ({mode}): {started.title}{suffix}"

    def stop(self) -> str:
        with self._lock:
            was_playing = (
                self._current is not None
                or self._ipc_path is not None
                or (self._process is not None and self._process.poll() is None)
            )
            self._stop_playback()
        if was_playing:
            return "YouTube 재생을 중지했습니다."
        return "재생 중인 YouTube가 없습니다."


_player = YouTubePlayer()


def play_youtube(query: str) -> str:
    normalized = _normalize_youtube_query(query)
    if normalized != query.strip():
        print(
            f"[youtube] 검색어 보정: {query!r} → {normalized!r}",
            file=sys.stderr,
        )
    return _player.play_query(normalized, replace=True)


def stop_youtube() -> str:
    return _player.stop()
