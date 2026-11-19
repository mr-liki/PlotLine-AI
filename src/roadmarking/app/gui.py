"""Plotline-AI desktop window (tkinter, part of the Python standard library).

Browse LAS/LAZ -> Save Output As -> Start Processing. The job runs in a worker thread; the
window stays responsive and shows the measured progress of the current stage. Cancel stops at
the next safe unit; nothing is written to the destination unless the output passed validation.
"""

from __future__ import annotations

import contextlib
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

TITLE = "Plotline-AI"
SUBTITLE = "Automatic LiDAR Road-Marking Digitization"
ASSETS = Path(__file__).resolve().parent / "assets"
BUTTON_WIDTH = 24


class App:
    def __init__(self, root: tk.Tk, pause_pid: int | None = None):
        self.root = root
        # Host program to pause while a job runs (given by the PlotLine plugin): the job gets
        # the computer's CPU and GPU time to itself; a watchdog resumes the host if
        # Plotline-AI ends unexpectedly.
        self.pause_pid = pause_pid
        self.host_paused = False
        if pause_pid:
            from roadmarking.app.host_pause import start_watchdog

            start_watchdog(pause_pid)
        self.events: queue.Queue = queue.Queue()
        self.cancel = threading.Event()
        self.worker: threading.Thread | None = None
        self.started = 0.0
        self.result_path: Path | None = None
        root.title(TITLE)
        root.geometry("860x700")
        root.minsize(760, 600)
        icon = ASSETS / "plotline_ai.ico"
        if icon.exists():
            with contextlib.suppress(tk.TclError):
                root.iconbitmap(default=str(icon))
        root.columnconfigure(0, weight=1)
        style = ttk.Style(root)
        with contextlib.suppress(tk.TclError):
            style.theme_use("vista")
        pad = dict(padx=14, pady=6)

        head = ttk.Frame(root)
        head.grid(row=0, column=0, sticky="ew", **pad)
        ttk.Label(head, text=TITLE, font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(head, text=SUBTITLE, font=("Segoe UI", 11)).pack(anchor="w")

        files = ttk.LabelFrame(root, text="Input and output")
        files.grid(row=1, column=0, sticky="ew", **pad)
        files.columnconfigure(1, weight=1)
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.output_auto = True  # output name suggested from the input (not chosen by the user)
        ttk.Button(
            files, text="Browse LAS/LAZ", command=self.browse_input, width=BUTTON_WIDTH
        ).grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(files, textvariable=self.input_var, state="readonly").grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Button(
            files, text="Add overlapping files", command=self.browse_extra, width=BUTTON_WIDTH
        ).grid(row=1, column=0, sticky="w", padx=8, pady=6)
        extra_row = ttk.Frame(files)
        extra_row.grid(row=1, column=1, sticky="ew", padx=8)
        extra_row.columnconfigure(0, weight=1)
        self.extra_label = ttk.Label(
            extra_row,
            text="Optional: other passes of the same road (both directions give full coverage)",
            foreground="#555",
        )
        self.extra_label.grid(row=0, column=0, sticky="w")
        self.clear_btn = ttk.Button(
            extra_row, text="Clear", command=self.clear_extra, width=8, state="disabled"
        )
        self.clear_btn.grid(row=0, column=1, sticky="e")
        ttk.Button(
            files, text="Save Output As", command=self.browse_output, width=BUTTON_WIDTH
        ).grid(row=2, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(files, textvariable=self.output_var, state="readonly").grid(
            row=2, column=1, sticky="ew", padx=8
        )
        self.extra: list[str] = []

        run = ttk.LabelFrame(root, text="Processing")
        run.grid(row=2, column=0, sticky="ew", **pad)
        run.columnconfigure(0, weight=1)
        buttons = ttk.Frame(run)
        buttons.grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.start_btn = ttk.Button(
            buttons, text="Start Processing", command=self.start, state="disabled"
        )
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(
            buttons, text="Cancel Processing", command=self.request_cancel, state="disabled"
        )
        self.cancel_btn.pack(side="left", padx=8)
        bar = ttk.Frame(run)
        bar.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        bar.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(bar, maximum=1000, mode="determinate")
        self.progress.grid(row=0, column=0, sticky="ew")
        self.percent_var = tk.StringVar(value="0 %")
        ttk.Label(bar, textvariable=self.percent_var, width=6, anchor="e").grid(
            row=0, column=1, padx=(8, 0)
        )
        self.stage_var = tk.StringVar(value="Select a LAS/LAZ file to begin.")
        ttk.Label(run, textvariable=self.stage_var).grid(row=2, column=0, sticky="w", padx=8)
        self.elapsed_var = tk.StringVar(value="")
        ttk.Label(run, textvariable=self.elapsed_var, foreground="#555").grid(
            row=3, column=0, sticky="w", padx=8, pady=(0, 6)
        )

        done = ttk.LabelFrame(root, text="Result")
        done.grid(row=3, column=0, sticky="nsew", **pad)
        root.rowconfigure(3, weight=1)
        done.columnconfigure(0, weight=1)
        done.rowconfigure(0, weight=1)
        self.result_text = tk.Text(
            done, height=10, wrap="word", state="disabled", relief="flat", font=("Segoe UI", 10)
        )
        self.result_text.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=6)
        scroll = ttk.Scrollbar(done, orient="vertical", command=self.result_text.yview)
        scroll.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=6)
        self.result_text.config(yscrollcommand=scroll.set)
        actions = ttk.Frame(done)
        actions.grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=6)
        self.open_btn = ttk.Button(
            actions, text="Open Output Folder", command=self.open_folder, state="disabled"
        )
        self.open_btn.pack(side="left")
        self.log_btn = ttk.Button(
            actions, text="Open Job Log", command=self.open_log, state="disabled"
        )
        self.log_btn.pack(side="left", padx=8)
        self.log_path: Path | None = None
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    # ------------------------------------------------------------ file selection
    def browse_input(self):
        path = filedialog.askopenfilename(
            title="Select LAS/LAZ point cloud",
            filetypes=[("LiDAR point cloud", "*.las *.laz"), ("All files", "*.*")],
        )
        if not path:
            return
        self.set_input(path)

    def set_input(self, path):
        """New input file: the output name follows it (unless the user chose one themself) and
        the previous job's progress and result are cleared, so a failed or finished job never
        blocks the next one."""
        p = Path(path)
        self.input_var.set(os.path.normpath(path))
        suggested = p.stem + "_digitized.gpkg"
        out = self.output_var.get()
        if not out or self.output_auto:
            folder = Path(out).parent if out else p.parent
            self.output_var.set(os.path.normpath(folder / suggested))
            self.output_auto = True
        self.reset_panel()
        self.refresh()

    def reset_panel(self):
        if self.worker is not None:
            return
        self.progress["value"] = 0
        self.percent_var.set("0 %")
        self.stage_var.set("Ready. Click Start Processing.")
        self.elapsed_var.set("")
        self.set_result("")
        self.result_path = None
        self.open_btn.config(state="disabled")

    def browse_extra(self):
        paths = filedialog.askopenfilenames(
            title="Select overlapping LAS/LAZ files (same survey)",
            filetypes=[("LiDAR point cloud", "*.las *.laz")],
        )
        if paths:
            self.extra = [os.path.normpath(p) for p in paths]
            names = ", ".join(Path(p).name for p in self.extra)
            self.extra_label.config(text=f"{len(self.extra)} overlapping file(s): {names}")
            self.clear_btn.config(state="normal")

    def clear_extra(self):
        self.extra = []
        self.extra_label.config(
            text="Optional: other passes of the same road (both directions give full coverage)"
        )
        self.clear_btn.config(state="disabled")

    def browse_output(self):
        initial = Path(self.output_var.get() or self.input_var.get() or Path.home())
        path = filedialog.asksaveasfilename(
            title="Save digitized GeoPackage as",
            defaultextension=".gpkg",
            initialdir=str(initial.parent if initial.suffix else initial),
            initialfile=initial.name if initial.suffix == ".gpkg" else "",
            filetypes=[("GeoPackage", "*.gpkg")],
            confirmoverwrite=False,
        )
        if path:
            if not path.lower().endswith(".gpkg"):
                path += ".gpkg"
            self.output_var.set(os.path.normpath(path))
            self.output_auto = False  # chosen by the user: keep it
            self.reset_panel()
            self.refresh()

    def refresh(self):
        ok = (
            self.worker is None
            and Path(self.input_var.get()).is_file()
            and self.output_var.get().lower().endswith(".gpkg")
        )
        self.start_btn.config(state="normal" if ok else "disabled")

    # ------------------------------------------------------------ processing
    def start(self):
        if self.worker is not None:
            return
        out = Path(self.output_var.get())
        overwrite = False
        if out.exists():
            if not messagebox.askyesno(
                TITLE,
                f"{out.name} already exists.\n\nReplace it with the new result?",
                parent=self.root,
            ):
                return
            overwrite = True
        self.cancel.clear()
        self.result_path = None
        self.set_result("")
        self.open_btn.config(state="disabled")
        self.log_btn.config(state="disabled")
        self.percent_var.set("0 %")
        self.start_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.progress["value"] = 0
        self.started = time.perf_counter()
        self.worker = threading.Thread(
            target=self.run_job,
            args=(self.input_var.get(), str(out), overwrite, list(self.extra)),
            daemon=True,
        )
        self.worker.start()
        self.pause_host()

    def run_job(self, src, out, overwrite, extra):
        from roadmarking.app.pipeline import Cancelled, JobError, run_job

        try:
            res = run_job(
                src,
                out,
                lambda stage, frac, msg: self.events.put(("progress", frac, msg)),
                self.cancel,
                overwrite=overwrite,
                extra_las=extra,
            )
            self.events.put(("done", res))
        except Cancelled:
            self.events.put(("cancelled",))
        except JobError as e:
            self.events.put(("error", str(e)))
        except Exception as e:
            self.events.put(("error", f"Unexpected error: {e}"))

    def request_cancel(self):
        if self.worker is not None:
            self.cancel.set()
            self.cancel_btn.config(state="disabled")
            self.stage_var.set("Cancelling at the next safe point...")

    def poll(self):
        try:
            while True:
                ev = self.events.get_nowait()
                kind = ev[0]
                if kind == "progress":
                    self.progress["value"] = int(1000 * ev[1])
                    self.percent_var.set(f"{int(100 * ev[1])} %")
                    self.stage_var.set(ev[2])
                elif kind == "done":
                    self.finish()
                    res = ev[1]
                    self.result_path = res.output
                    self.log_path = res.report
                    self.progress["value"] = 1000
                    self.percent_var.set("100 %")
                    self.stage_var.set("Completed")
                    minutes, seconds = divmod(int(res.seconds), 60)
                    text = (
                        "Digitization completed successfully.\n\n"
                        f"Output: {res.output}\n"
                        f"Features: {res.lines} lines (layer Highway), "
                        f"{res.polygons} polygons (layer Other Marks)\n"
                        f"Coordinate reference: {res.crs}\n"
                        f"Processing time: {minutes} min {seconds} s\n"
                    )
                    if res.warnings:
                        text += "\nNotes:\n- " + "\n- ".join(res.warnings)
                    self.set_result(text)
                    self.open_btn.config(state="normal")
                    self.log_btn.config(state="normal" if res.report else "disabled")
                elif kind == "cancelled":
                    self.finish()
                    self.stage_var.set("Cancelled. No output was written.")
                    self.set_result("Processing was cancelled. The input file was not changed.")
                elif kind == "error":
                    self.finish()
                    self.stage_var.set("Failed. Select another file (or fix this one) and Start.")
                    self.set_result(
                        f"Processing failed.\n\n{ev[1]}\n\n"
                        "Nothing was written. Select another LAS/LAZ file and start again."
                    )
                    self.log_path = latest_log()
                    self.log_btn.config(state="normal" if self.log_path else "disabled")
                    # no modal dialog: it can open behind the window and block it
                    self.root.bell()
                    self.root.lift()
        except queue.Empty:
            pass
        if self.worker is not None:
            seconds = int(time.perf_counter() - self.started)
            self.elapsed_var.set(f"Elapsed: {seconds // 60} min {seconds % 60:02d} s")
        self.root.after(200, self.poll)

    def pause_host(self):
        if self.pause_pid and not self.host_paused:
            from roadmarking.app.host_pause import pause

            self.host_paused = pause(self.pause_pid)
            if self.host_paused:
                self.stage_var.set(
                    "Processing; NakshaAI-LiDAR is paused until the job ends (full resources)."
                )

    def resume_host(self):
        if self.host_paused:
            from roadmarking.app.host_pause import resume

            resume(self.pause_pid)
            self.host_paused = False

    def finish(self):
        self.resume_host()
        self.worker = None
        self.cancel.clear()
        self.cancel_btn.config(state="disabled")
        self.refresh()

    def set_result(self, text):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.config(state="disabled")

    def open_log(self):
        if self.log_path is not None and Path(self.log_path).exists():
            os.startfile(str(self.log_path))  # opens the job log in the default viewer

    def open_folder(self):
        if self.result_path is not None:
            subprocess.Popen(["explorer", "/select,", str(self.result_path)])

    def close(self):
        if self.worker is not None:
            if not messagebox.askyesno(
                TITLE, "Processing is running. Cancel it and exit?", parent=self.root
            ):
                return
            self.cancel.set()
        self.resume_host()
        self.root.destroy()


def latest_log():
    from roadmarking.app.pipeline import app_home

    logs = sorted((app_home() / "logs").glob("*.json"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def main(argv=None):
    import sys

    argv = sys.argv[1:] if argv is None else argv
    pause_pid = None
    if "--pause-pid" in argv:
        with contextlib.suppress(ValueError, IndexError):
            pause_pid = int(argv[argv.index("--pause-pid") + 1])
    with contextlib.suppress(Exception):  # own taskbar icon rather than the Python one
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Plotline-AI.Desktop")
    root = tk.Tk()
    app = App(root, pause_pid=pause_pid)
    try:
        root.mainloop()
    finally:
        app.resume_host()  # never leave the host paused


if __name__ == "__main__":
    main()
