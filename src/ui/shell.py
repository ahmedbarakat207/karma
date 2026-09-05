"""Interactive shell sessions for the dashboard Shell tab.

Each session is a real login-less `bash` on the robot (same user the robot
runs as), attached to a PTY so readline, colors, and fullscreen programs
behave. All I/O runs off the asyncio loop: a daemon reader thread pushes
raw bytes out, writes go straight to the PTY master fd.

Security: sessions are only ever created by the authenticated `/api/shell`
websocket in `server.py` (same session cookie + rate-limited login as the
rest of the dashboard). `SHELL_ENABLED=0` disables the endpoint entirely.
"""

import os
import pty
import signal
import subprocess
import threading
import time
from typing import Callable, Optional

from src import config

READ_CHUNK = 65536


def _pick_shell() -> list:
    for candidate in (os.environ.get("SHELL", ""), "/bin/bash", "/bin/sh"):
        if candidate and os.path.exists(candidate):
            return [candidate, "-i"] if candidate.endswith("bash") else [candidate]
    return ["/bin/sh"]


class ShellSession:
    """One PTY-backed shell. `on_output(bytes|None)` fires on reader thread."""

    def __init__(self, on_output: Callable[[Optional[bytes]], None],
                 cols: int = 100, rows: int = 30):
        self._on_output = on_output
        self._closed = threading.Event()
        self.last_io = time.time()
        self.master_fd: Optional[int] = None
        self.proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None

        master, slave = pty.openpty()
        self._resize_fd(master, rows, cols)
        env = dict(os.environ)
        env["TERM"] = "xterm-256color"
        try:
            self.proc = subprocess.Popen(
                _pick_shell(),
                stdin=slave, stdout=slave, stderr=slave,
                cwd=getattr(config, "BASE_DIR", os.getcwd()),
                env=env, close_fds=True,
                start_new_session=True,
            )
        except Exception:
            os.close(master)
            os.close(slave)
            raise
        os.close(slave)
        self.master_fd = master
        self._reader = threading.Thread(target=self._read_loop,
                                        daemon=True, name="shell-reader")
        self._reader.start()

    @staticmethod
    def _resize_fd(fd: int, rows: int, cols: int) -> None:
        try:
            import fcntl
            import struct
            import termios
            fcntl.ioctl(fd, termios.TIOCSWINSZ,
                        struct.pack("HHHH", max(1, rows), max(1, cols), 0, 0))
        except Exception:
            pass

    def resize(self, rows: int, cols: int) -> None:
        if self.master_fd is not None and not self._closed.is_set():
            self._resize_fd(self.master_fd, rows, cols)

    def write(self, data: str) -> bool:
        if self._closed.is_set() or self.master_fd is None:
            return False
        try:
            os.write(self.master_fd, data.encode("utf-8", "replace"))
            self.last_io = time.time()
            return True
        except OSError:
            return False

    def _read_loop(self) -> None:
        try:
            while not self._closed.is_set():
                try:
                    chunk = os.read(self.master_fd, READ_CHUNK)
                except OSError:
                    break
                if not chunk:
                    break
                self.last_io = time.time()
                try:
                    self._on_output(chunk)
                except Exception:
                    break
        finally:
            try:
                self._on_output(None)  # EOF sentinel: process exited
            except Exception:
                pass

    def exit_code(self) -> Optional[int]:
        try:
            return self.proc.poll() if self.proc else None
        except Exception:
            return None

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        proc, fd = self.proc, self.master_fd
        self.proc, self.master_fd = None, None
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
