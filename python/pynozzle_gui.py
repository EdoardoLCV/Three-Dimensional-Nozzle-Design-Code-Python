#!/usr/bin/env python3
"""Graphical front-end for the pynozzle suite.

A small Tkinter application that brings back a clickable interface for the
three ported tools (MOC_Grid_BDE / STT2001 / 3D_MOC). It calls the same
solver code as the command-line tools, so results are identical.

Run it with::

    python pynozzle_gui.py

Tkinter ships with CPython. On Debian/Ubuntu, if you see
``ModuleNotFoundError: No module named 'tkinter'`` install it with::

    sudo apt install python3-tk
"""
from __future__ import annotations

import io
import queue
import sys
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# The three CLI entry points. Importing here (rather than inside the worker)
# surfaces a missing-install error immediately and clearly.
from pynozzle.moc2d.cli import main as moc2d_main
from pynozzle.stt.cli import main as stt_main
from pynozzle.moc3d.cli import main as moc3d_main


# --------------------------------------------------------------------------
#  A stdout/stderr sink that pushes text onto a thread-safe queue. The Tk
#  main loop drains the queue with .after(), so the worker thread never
#  touches a widget directly (Tkinter is not thread-safe).
# --------------------------------------------------------------------------
class _QueueWriter(io.TextIOBase):
    def __init__(self, q: "queue.Queue[str]"):
        self._q = q

    def write(self, s: str) -> int:
        if s:
            self._q.put(s)
        return len(s)

    def flush(self) -> None:  # pragma: no cover - nothing buffered
        pass


class PynozzleGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("pynozzle - Nozzle Design Suite")
        self.geometry("780x640")
        self.minsize(680, 560)

        self._log_q: "queue.Queue[str]" = queue.Queue()
        self._busy = False
        self._run_buttons: list[ttk.Button] = []

        self._build_widgets()
        self.after(80, self._drain_log)

    # ---- layout ----------------------------------------------------------
    def _build_widgets(self) -> None:
        nb = ttk.Notebook(self)
        nb.pack(side="top", fill="x", padx=8, pady=(8, 4))

        self._build_moc2d_tab(nb)
        self._build_stt_tab(nb)
        self._build_moc3d_tab(nb)

        # shared status + log
        self._status = tk.StringVar(value="Ready.")
        status_lbl = ttk.Label(self, textvariable=self._status, anchor="w",
                               relief="sunken", padding=(6, 2))
        status_lbl.pack(side="top", fill="x", padx=8)

        log_frame = ttk.LabelFrame(self, text="Output")
        log_frame.pack(side="top", fill="both", expand=True, padx=8, pady=6)
        self._log = tk.Text(log_frame, wrap="word", height=14,
                            background="#111418", foreground="#d6deeb",
                            insertbackground="#d6deeb")
        yscroll = ttk.Scrollbar(log_frame, command=self._log.yview)
        self._log.configure(yscrollcommand=yscroll.set, state="disabled")
        yscroll.pack(side="right", fill="y")
        self._log.pack(side="left", fill="both", expand=True)

        btnbar = ttk.Frame(self)
        btnbar.pack(side="bottom", fill="x", padx=8, pady=(0, 8))
        ttk.Button(btnbar, text="Clear log", command=self._clear_log
                   ).pack(side="right")

    # ---- small helpers for building forms --------------------------------
    def _file_row(self, parent, label, var, *, patterns, row, save=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w",
                                            padx=4, pady=3)
        ttk.Entry(parent, textvariable=var, width=58).grid(
            row=row, column=1, sticky="we", padx=4, pady=3)

        def browse():
            initial = var.get() or "."
            path = filedialog.askopenfilename(
                title=label, initialdir=str(Path(initial).parent),
                filetypes=patterns)
            if path:
                var.set(path)
                self._after_input_pick(path)

        ttk.Button(parent, text="Browse...", command=browse).grid(
            row=row, column=2, padx=4, pady=3)

    def _dir_row(self, parent, label, var, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w",
                                            padx=4, pady=3)
        ttk.Entry(parent, textvariable=var, width=58).grid(
            row=row, column=1, sticky="we", padx=4, pady=3)

        def browse():
            path = filedialog.askdirectory(
                title=label, initialdir=var.get() or ".")
            if path:
                var.set(path)

        ttk.Button(parent, text="Browse...", command=browse).grid(
            row=row, column=2, padx=4, pady=3)

    def _after_input_pick(self, path: str) -> None:
        """When the user picks an input file, default the output dir to its
        folder (and, for STT, the MOC-input dir too) if those are empty."""
        folder = str(Path(path).parent)
        if not self._moc2d_out.get():
            self._moc2d_out.set(folder)
        if not self._stt_out.get():
            self._stt_out.set(folder)
        if not self._stt_in.get():
            self._stt_in.set(folder)
        if not self._moc3d_out.get():
            self._moc3d_out.set(folder)

    # ---- MOC2D tab -------------------------------------------------------
    def _build_moc2d_tab(self, nb) -> None:
        tab = ttk.Frame(nb)
        nb.add(tab, text="2D MOC  (MOC_Grid_BDE)")
        tab.columnconfigure(1, weight=1)

        self._moc2d_inp = tk.StringVar()
        self._moc2d_out = tk.StringVar()
        self._moc2d_summary_only = tk.BooleanVar(value=False)

        self._file_row(tab, ".inp file", self._moc2d_inp,
                       patterns=[("MOC input", "*.inp"), ("All files", "*.*")],
                       row=0)
        self._dir_row(tab, "Output folder", self._moc2d_out, row=1)
        ttk.Checkbutton(tab, text="Summary only (no grid files)",
                        variable=self._moc2d_summary_only).grid(
            row=2, column=1, sticky="w", padx=4, pady=3)

        b = ttk.Button(tab, text="Run 2D MOC",
                       command=self._run_moc2d)
        b.grid(row=3, column=1, sticky="e", padx=4, pady=8)
        self._run_buttons.append(b)

    def _run_moc2d(self) -> None:
        inp = self._moc2d_inp.get().strip()
        out = self._moc2d_out.get().strip() or "."
        if not inp:
            messagebox.showwarning("Missing input", "Choose a .inp file first.")
            return
        argv = [inp, "-o", out]
        if self._moc2d_summary_only.get():
            argv.append("--summary-only")
        self._launch(moc2d_main, argv, "2D MOC")

    # ---- STT tab ---------------------------------------------------------
    def _build_stt_tab(self, nb) -> None:
        tab = ttk.Frame(nb)
        nb.add(tab, text="Streamline Trace  (STT2001)")
        tab.columnconfigure(1, weight=1)

        self._stt_inp = tk.StringVar()
        self._stt_in = tk.StringVar()
        self._stt_out = tk.StringVar()

        self._file_row(tab, ".inp file", self._stt_inp,
                       patterns=[("STT input", "*.inp"), ("All files", "*.*")],
                       row=0)
        self._dir_row(tab, "MOC input folder", self._stt_in, row=1)
        self._dir_row(tab, "Output folder", self._stt_out, row=2)
        ttk.Label(tab, foreground="#888",
                  text="MOC input folder must contain MOC_SL.plt, MOC_Grid.plt, "
                       "summary.out\nand the friction table from the prior 2D run.").grid(
            row=3, column=1, sticky="w", padx=4, pady=(0, 4))

        b = ttk.Button(tab, text="Run Streamline Trace",
                       command=self._run_stt)
        b.grid(row=4, column=1, sticky="e", padx=4, pady=8)
        self._run_buttons.append(b)

    def _run_stt(self) -> None:
        inp = self._stt_inp.get().strip()
        out = self._stt_out.get().strip() or "."
        if not inp:
            messagebox.showwarning("Missing input", "Choose a .inp file first.")
            return
        argv = [inp, "-o", out]
        indir = self._stt_in.get().strip()
        if indir:
            argv += ["-i", indir]
        self._launch(stt_main, argv, "Streamline Trace")

    # ---- MOC3D tab -------------------------------------------------------
    def _build_moc3d_tab(self, nb) -> None:
        tab = ttk.Frame(nb)
        nb.add(tab, text="3D MOC  (3D_MOC)")
        tab.columnconfigure(1, weight=1)

        self._moc3d_geo = tk.StringVar()
        self._moc3d_out = tk.StringVar()
        self._file_row(tab, ".geo file", self._moc3d_geo,
                       patterns=[("Wall contour", "*.geo"), ("All files", "*.*")],
                       row=0)
        self._dir_row(tab, "Output folder", self._moc3d_out, row=1)

        # numeric parameters (label, StringVar, default)
        params = ttk.LabelFrame(tab, text="Chamber / solver parameters")
        params.grid(row=2, column=0, columnspan=3, sticky="we", padx=4, pady=6)
        for c in (1, 3):
            params.columnconfigure(c, weight=1)

        self._moc3d_vars: dict[str, tk.StringVar] = {}
        fields = [
            ("--p0", "Chamber pressure p0 (psia)", "1000"),
            ("--t0", "Chamber temperature T0 (R)", "530"),
            ("--mach0", "Initial Mach", "1.1"),
            ("--mol-wt", "Molecular weight", "28.96"),
            ("--gamma", "Gamma", "1.4"),
            ("--theta0", "Theta0 (deg)", "0"),
            ("--psi0", "Psi0 (deg)", "0"),
            ("--n-div", "Angular divisions", "36"),
            ("--x-step", "X output stride", "1"),
            ("--y-step", "Y output stride", "1"),
            ("--z-step", "Z output stride", "10"),
            ("--step-step", "Intermediate stride", "999"),
        ]
        for i, (flag, label, default) in enumerate(fields):
            r, c = divmod(i, 2)
            var = tk.StringVar(value=default)
            self._moc3d_vars[flag] = var
            ttk.Label(params, text=label).grid(
                row=r, column=c * 2, sticky="w", padx=4, pady=2)
            ttk.Entry(params, textvariable=var, width=12).grid(
                row=r, column=c * 2 + 1, sticky="w", padx=4, pady=2)

        ttk.Label(tab, text="Surface fit").grid(row=3, column=0, sticky="w",
                                                 padx=4, pady=3)
        self._moc3d_fit = tk.StringVar(value="All Point Spline")
        ttk.Combobox(tab, textvariable=self._moc3d_fit, state="readonly",
                     values=("All Point Spline", "9 Point Spline"), width=20
                     ).grid(row=3, column=1, sticky="w", padx=4, pady=3)

        ttk.Label(tab, foreground="#888",
                  text="Tip: set Z output stride = 1 to write every plane "
                       "(matches the sample outputs). The full cone10 march "
                       "takes ~1 minute.").grid(
            row=4, column=1, sticky="w", padx=4, pady=(0, 4))

        b = ttk.Button(tab, text="Run 3D MOC", command=self._run_moc3d)
        b.grid(row=5, column=1, sticky="e", padx=4, pady=8)
        self._run_buttons.append(b)

    def _run_moc3d(self) -> None:
        geo = self._moc3d_geo.get().strip()
        out = self._moc3d_out.get().strip() or "."
        if not geo:
            messagebox.showwarning("Missing input", "Choose a .geo file first.")
            return
        argv = [geo, "-o", out]
        for flag, var in self._moc3d_vars.items():
            val = var.get().strip()
            if val:
                argv += [flag, val]
        argv += ["--surface-fit", self._moc3d_fit.get()]
        self._launch(moc3d_main, argv, "3D MOC")

    # ---- run machinery ---------------------------------------------------
    def _launch(self, func, argv, label) -> None:
        if self._busy:
            messagebox.showinfo("Busy", "A run is already in progress.")
            return
        self._busy = True
        for b in self._run_buttons:
            b.state(["disabled"])
        self._status.set(f"Running {label}...")
        self._append_log(f"\n=== {label}: {' '.join(argv)} ===\n")

        def worker():
            writer = _QueueWriter(self._log_q)
            rc = None
            try:
                with redirect_stdout(writer), redirect_stderr(writer):
                    rc = func(list(argv))
            except SystemExit as e:  # _build_calc raises this on bad inputs
                self._log_q.put(f"\n[stopped] {e}\n")
                rc = e.code if isinstance(e.code, int) else 1
            except Exception:
                self._log_q.put("\n[error]\n" + traceback.format_exc() + "\n")
                rc = 1
            self._log_q.put(("__DONE__", label, rc))

        threading.Thread(target=worker, daemon=True).start()

    def _drain_log(self) -> None:
        try:
            while True:
                item = self._log_q.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "__DONE__":
                    _, label, rc = item
                    self._finish(label, rc)
                else:
                    self._append_log(item)
        except queue.Empty:
            pass
        self.after(80, self._drain_log)

    def _finish(self, label, rc) -> None:
        self._busy = False
        for b in self._run_buttons:
            b.state(["!disabled"])
        if rc == 0:
            self._status.set(f"{label} finished successfully.")
        else:
            self._status.set(f"{label} failed (exit code {rc}). See output.")

    # ---- log widget helpers ---------------------------------------------
    def _append_log(self, text: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", text)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _clear_log(self) -> None:
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")


def main() -> int:
    app = PynozzleGUI()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
