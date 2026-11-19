"""PlotLine - NakshaAI-LiDAR plugin that starts Plotline-AI (automatic LiDAR road-marking
digitization) as its own, independent program.

- The plugin only adds a "PlotLine" button to the Plugins ribbon. Nothing of Plotline-AI is
  imported into the host: it runs in its own process with its own bundled Python runtime
  (``app/runtime``), so the host and Plotline-AI never share libraries, memory or threads.
- While a Plotline-AI job runs, the host is paused (its process is suspended) so the job gets
  the computer's full CPU and GPU time; it is resumed when the job ends, is cancelled, fails,
  or if Plotline-AI is closed or crashes (a watchdog process guarantees the resume).
- No file of the host software is changed.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
APP_DIR = PLUGIN_DIR / "app"
LABEL = "PlotLine"

# Variables of the host's own (frozen) Python/Qt/GDAL that must not leak into Plotline-AI.
_HOST_ENV = (
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "QT_PLUGIN_PATH",
    "QT_QPA_PLATFORM_PLUGIN_PATH",
    "GDAL_DATA",
    "GDAL_DRIVER_PATH",
    "PROJ_LIB",
    "PROJ_DATA",
    "TCL_LIBRARY",
    "TK_LIBRARY",
    "_MEIPASS2",
)


class PlotLinePlugin:
    def __init__(self):
        self.app_window = None
        self.process = None
        self._icon_timer = None

    # ------------------------------------------------------------------ host interface
    def on_load(self, app_window):
        self.app_window = app_window
        try:
            from PySide6.QtCore import QTimer

            # The host builds the ribbon after loading plugins (and rebuilds it later): put the
            # PlotLine icon on our button whenever it (re)appears.
            self._icon_timer = QTimer()
            self._icon_timer.setInterval(1500)
            self._icon_timer.timeout.connect(self._apply_icon)
            self._icon_timer.start()
            QTimer.singleShot(300, self._apply_icon)
        except Exception:
            self._icon_timer = None

    def get_ribbon_button(self):
        return {"label": LABEL, "emoji": "", "callback": self.run, "toggleable": False}

    def on_unload(self):
        if self._icon_timer is not None:
            with contextlib.suppress(Exception):
                self._icon_timer.stop()
            self._icon_timer = None

    # ------------------------------------------------------------------ actions
    def run(self, *args):
        if self.process is not None and self.process.poll() is None:
            self._message("Plotline-AI is already open.")
            return
        pythonw = APP_DIR / "runtime" / "pythonw.exe"
        launcher = APP_DIR / "launch.pyw"
        if not pythonw.exists() or not launcher.exists():
            self._message(
                "The PlotLine plugin is incomplete (app/runtime is missing). "
                "Reinstall it from the PlotLine plugin ZIP."
            )
            return
        env = {k: v for k, v in os.environ.items() if k.upper() not in _HOST_ENV}
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            self.process = subprocess.Popen(
                [str(pythonw), str(launcher), "--pause-pid", str(os.getpid())],
                cwd=str(APP_DIR),
                env=env,
                creationflags=flags,
                close_fds=True,
            )
        except Exception as exc:
            self._message(f"Plotline-AI could not be started: {exc}")

    # ------------------------------------------------------------------ helpers
    def _apply_icon(self):
        try:
            from PySide6.QtCore import QSize
            from PySide6.QtGui import QIcon
            from PySide6.QtWidgets import QApplication, QPushButton

            icon_file = PLUGIN_DIR / "icon.png"
            if not icon_file.exists():
                return
            icon = QIcon(str(icon_file))
            for widget in QApplication.topLevelWidgets():
                for btn in widget.findChildren(QPushButton, "ribbonButton"):
                    if btn.property("ribbonText") != LABEL:
                        continue
                    if btn.property("plotlineIcon"):
                        continue
                    btn.setText("")
                    btn.setIcon(icon)
                    btn.setIconSize(QSize(28, 28))
                    btn.setToolTip("PlotLine - automatic road-marking digitization (Plotline-AI)")
                    btn.setProperty("plotlineIcon", True)
        except Exception:
            pass

    def _message(self, text):
        try:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.information(self.app_window, LABEL, text)
        except Exception:
            print(f"[PlotLine] {text}")
