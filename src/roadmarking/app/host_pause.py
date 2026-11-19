"""Pause the host program (e.g. the NakshaAI-LiDAR GIS that launched Plotline-AI as a plugin)
while a job runs, so the job has the computer's CPU and GPU time to itself.

- ``pause(pid)`` suspends every thread of the host process; ``resume(pid)`` lets it continue
  exactly where it was. The host's window stays on screen but does not respond while paused.
- A watchdog process (``python -m roadmarking.app.host_pause --watch <plotline pid> <host pid>``)
  resumes the host if Plotline-AI exits or crashes while the host is paused, so the host can
  never be left frozen.
- Only the process given by the plugin (``--pause-pid``) is ever paused, never anything else.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time


def _proc(pid):
    import psutil

    try:
        return psutil.Process(int(pid))
    except (psutil.Error, ValueError, TypeError):
        return None


def pause(pid) -> bool:
    p = _proc(pid)
    if p is None or p.pid == os.getpid():
        return False
    try:
        p.suspend()
        return True
    except Exception:
        return False


def resume(pid) -> bool:
    p = _proc(pid)
    if p is None:
        return False
    try:
        p.resume()
        return True
    except Exception:
        return False


def start_watchdog(host_pid):
    """Detached helper that resumes ``host_pid`` once this Plotline-AI process has ended."""
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    exe = sys.executable
    if exe.lower().endswith("pythonw.exe"):
        console = exe[: -len("pythonw.exe")] + "python.exe"
        exe = console if os.path.exists(console) else exe
    with contextlib.suppress(Exception):
        subprocess.Popen(
            [exe, "-m", "roadmarking.app.host_pause", "--watch", str(os.getpid()), str(host_pid)],
            creationflags=flags,
            close_fds=True,
        )


def _watch(plotline_pid, host_pid):
    import psutil

    while True:
        if not psutil.pid_exists(plotline_pid):
            resume(host_pid)
            return
        if not psutil.pid_exists(host_pid):
            return
        time.sleep(1.0)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--watch":
        _watch(int(sys.argv[2]), int(sys.argv[3]))
