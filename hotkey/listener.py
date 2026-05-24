import threading

from pynput import keyboard


class HotkeyListener:
    """Option+Space 글로벌 단축키로 콜백을 호출한다."""

    def __init__(self, callback, combo: str = "<alt>+<space>"):
        self._callback = callback
        self._combo = combo
        self._listener: keyboard.GlobalHotKeys | None = None
        self._lock = threading.Lock()
        self._triggered = False

    def _on_activate(self):
        with self._lock:
            if self._triggered:
                return
            self._triggered = True
        try:
            self._callback()
        except KeyboardInterrupt:
            print("\n세션 중단")
        except Exception as error:
            print(f"\n[hotkey] 세션 오류: {error}")
        finally:
            with self._lock:
                self._triggered = False

    def start(self):
        self._listener = keyboard.GlobalHotKeys({self._combo: self._on_activate})
        self._listener.start()

    def stop(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
