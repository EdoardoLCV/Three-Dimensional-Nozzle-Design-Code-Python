#!/usr/bin/env python3
"""Graphical front-end for the pynozzle suite.

Recreates the three original MFC dialogs from the JHU/APL tool as closely
as the data model allows:

  * 2D MOC  (MOC_Grid_BDE)  - Nozzle Geometry / Type / Design Parameter /
    Throat Geometry / Flow Properties / Output Streamlines / Print Option /
    MOC Limiters, with a nozzle-contour plot of the result.
  * Streamline Trace (STT2001) - File inputs, Other inputs, SL flow-field
    centre sweep, five trimming/throat slots, and the computed nozzle
    parameter outputs.
  * 3D MOC (3D_MOC) - File I/O, Initial Plane Properties, Grid Setup,
    Print Output Parameters, Surface Fit.

Every tab lets you type the inputs by hand (like the original GUI) and
also Load / Save the matching ``.inp`` file. Plots use matplotlib if it is
installed; otherwise the rest of the program still works.

Run with::

    python pynozzle_gui.py

If you see ``No module named 'tkinter'`` install it:
    Ubuntu/Debian: sudo apt install python3-tk
    conda:         conda install tk
"""
from __future__ import annotations

import io
import queue
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# --- solver / IO imports (fail loudly if the package isn't installed) -------
from pynozzle.moc2d.inp import (
    MOCInput, read_inp as read_moc2d_inp, write_inp as write_moc2d_inp,
)
from pynozzle.moc2d.cli import _build_calc
from pynozzle.moc2d.io_writers import write_all as moc2d_write_all
from pynozzle.stt.inp import (
    STTInput, ConstraintSlot, read_inp as read_stt_inp,
    write_inp as write_stt_inp,
)
from pynozzle.stt.solver import STTSolver
from pynozzle.moc3d.geo import GeoConfig, read_geo
from pynozzle.moc3d.solver import MOC3DGrid

# --- optional plotting ------------------------------------------------------
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    _HAVE_MPL = True
except Exception:                       # pragma: no cover
    _HAVE_MPL = False


def _f(var, default=0.0):
    """Float from a tk var/string, tolerant of blanks."""
    try:
        return float(str(var).strip())
    except (ValueError, AttributeError):
        return default


def _i(var, default=0):
    try:
        return int(float(str(var).strip()))
    except (ValueError, AttributeError):
        return default


class _QueueWriter(io.TextIOBase):
    """stdout/stderr sink -> thread-safe queue drained by the Tk loop."""

    def __init__(self, q):
        self._q = q

    def write(self, s):
        if s:
            self._q.put(s)
        return len(s)


class PynozzleGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("pynozzle - Nozzle Design Suite")
        self.geometry("980x720")
        self.minsize(820, 600)

        self._log_q = queue.Queue()
        self._busy = False
        self._run_buttons = []

        nb = ttk.Notebook(self)
        nb.pack(side="top", fill="both", expand=True, padx=8, pady=(8, 4))
        self._build_moc2d_tab(nb)
        self._build_stt_tab(nb)
        self._build_moc3d_tab(nb)

        self._status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self._status, anchor="w",
                  relief="sunken", padding=(6, 2)).pack(side="top", fill="x",
                                                        padx=8)

        logf = ttk.LabelFrame(self, text="Output log")
        logf.pack(side="bottom", fill="both", expand=False, padx=8, pady=6)
        self._log = tk.Text(logf, wrap="word", height=8, background="#111418",
                            foreground="#d6deeb", insertbackground="#d6deeb")
        ys = ttk.Scrollbar(logf, command=self._log.yview)
        self._log.configure(yscrollcommand=ys.set, state="disabled")
        ys.pack(side="right", fill="y")
        self._log.pack(side="left", fill="both", expand=True)
        ttk.Button(logf, text="Clear", command=self._clear_log).pack(
            side="bottom", anchor="e", pady=2)

        self.after(80, self._drain_log)

    # ===================================================================
    #  small form helpers
    # ===================================================================
    def _labeled_entry(self, parent, label, default, width=10):
        row = parent.grid_size()[1]
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w",
                                            padx=4, pady=2)
        var = tk.StringVar(value=str(default))
        ttk.Entry(parent, textvariable=var, width=width).grid(
            row=row, column=1, sticky="w", padx=4, pady=2)
        return var

    def _grid_entry(self, parent, label, default, r, c, width=10):
        ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w",
                                           padx=4, pady=2)
        var = tk.StringVar(value=str(default))
        ttk.Entry(parent, textvariable=var, width=width).grid(
            row=r, column=c + 1, sticky="w", padx=4, pady=2)
        return var

    def _file_picker(self, parent, label, var, patterns, r, c=0):
        ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w",
                                           padx=4, pady=2)
        ttk.Entry(parent, textvariable=var, width=30).grid(
            row=r, column=c + 1, sticky="we", padx=4, pady=2)

        def browse():
            p = filedialog.askopenfilename(title=label, filetypes=patterns)
            if p:
                var.set(p)

        ttk.Button(parent, text="...", width=3, command=browse).grid(
            row=r, column=c + 2, padx=2, pady=2)

    def _dir_picker(self, parent, label, var, r, c=0):
        ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w",
                                           padx=4, pady=2)
        ttk.Entry(parent, textvariable=var, width=30).grid(
            row=r, column=c + 1, sticky="we", padx=4, pady=2)

        def browse():
            p = filedialog.askdirectory(title=label)
            if p:
                var.set(p)

        ttk.Button(parent, text="...", width=3, command=browse).grid(
            row=r, column=c + 2, padx=2, pady=2)

    def _plot_area(self, parent):
        """Return (frame, draw_fn). draw_fn(plot_callable) renders on the
        embedded matplotlib canvas, or shows a note if matplotlib is absent."""
        frame = ttk.LabelFrame(parent, text="Plot")
        if not _HAVE_MPL:
            ttk.Label(frame, foreground="#888", padding=20,
                      text="matplotlib not installed - plots disabled.\n"
                           "Install it with:  pip install matplotlib").pack(
                expand=True)
            return frame, (lambda fn: None)

        fig = Figure(figsize=(5, 3.2), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        def draw(plot_callable):
            fig.clear()
            ax = fig.add_subplot(111)
            try:
                plot_callable(ax)
            except Exception as e:        # pragma: no cover
                ax.text(0.5, 0.5, f"plot error: {e}", ha="center")
            fig.tight_layout()
            canvas.draw()

        return frame, draw

    # ===================================================================
    #  2D MOC tab
    # ===================================================================
    def _build_moc2d_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="2D MOC  (MOC_Grid_BDE)")

        left = ttk.Frame(tab)
        left.pack(side="left", fill="y", padx=4, pady=4)
        right = ttk.Frame(tab)
        right.pack(side="right", fill="both", expand=True, padx=4, pady=4)

        v = {}
        # --- Nozzle Geometry ---
        g_geom = ttk.LabelFrame(left, text="Nozzle Geometry")
        g_geom.grid(row=0, column=0, sticky="we", padx=4, pady=3)
        v["geom"] = tk.StringVar(value="axi")
        ttk.Radiobutton(g_geom, text="Axisymmetric", variable=v["geom"],
                        value="axi").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(g_geom, text="Planar", variable=v["geom"],
                        value="2d").grid(row=1, column=0, sticky="w")

        # --- Nozzle Type ---
        g_type = ttk.LabelFrame(left, text="Nozzle Type")
        g_type.grid(row=1, column=0, sticky="we", padx=4, pady=3)
        v["type"] = tk.StringVar(value="perfect")
        for i, (txt, val) in enumerate([("Perfect", "perfect"),
                                        ("RAO (Optimum Thrust)", "rao"),
                                        ("Set End Point", "endpoint"),
                                        ("Cone / Wedge", "cone")]):
            ttk.Radiobutton(g_type, text=txt, variable=v["type"],
                            value=val).grid(row=i, column=0, sticky="w",
                                            columnspan=4)
        v["x_e"] = self._grid_entry(g_type, "XEnd/R*", 4.308, 4, 0, 8)
        v["r_e"] = self._grid_entry(g_type, "REnd/R*", 1.6, 4, 2, 8)
        v["cone_angle"] = self._grid_entry(g_type, "Half Angle (deg)",
                                           15.0, 5, 0, 8)

        # --- Design Parameter ---
        g_dp = ttk.LabelFrame(left, text="Design Parameter")
        g_dp.grid(row=2, column=0, sticky="we", padx=4, pady=3)
        v["dp"] = tk.StringVar(value="mach")
        rows = [("Exit Mach", "mach", 4.0), ("Area Ratio", "eps", 15.0),
                ("Xend/R*", "length", 5.03258),
                ("Ptotal/Pexit", "pexit", 13.691)]
        v["dp_val"] = {}
        for i, (txt, val, dflt) in enumerate(rows):
            ttk.Radiobutton(g_dp, text=txt, variable=v["dp"],
                            value=val).grid(row=i, column=0, sticky="w")
            sv = tk.StringVar(value=str(dflt))
            ttk.Entry(g_dp, textvariable=sv, width=10).grid(
                row=i, column=1, padx=4, pady=1)
            v["dp_val"][val] = sv

        # --- Throat Geometry ---
        g_thr = ttk.LabelFrame(left, text="Throat Geometry")
        g_thr.grid(row=3, column=0, sticky="we", padx=4, pady=3)
        v["rwt_u"] = self._labeled_entry(g_thr, "UpStream Radius/R*", 1.0)
        v["rwt_d"] = self._labeled_entry(g_thr, "DownStream Radius/R*", 1.0)

        # --- Flow Properties ---
        g_flow = ttk.LabelFrame(right, text="Flow Properties")
        g_flow.grid(row=0, column=0, sticky="we", padx=4, pady=3)
        v["throat"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(g_flow, text="Throat Conditions?",
                        variable=v["throat"]).grid(row=0, column=0,
                                                   columnspan=2, sticky="w")
        v["pres_i"] = self._labeled_entry(g_flow, "Pressure (psia)", 1000.0)
        v["temp_i"] = self._labeled_entry(g_flow, "Temperature (R)", 530.0)
        v["mol_wt_i"] = self._labeled_entry(g_flow, "Mol. Wt", 28.96)
        v["gamma_i"] = self._labeled_entry(g_flow, "Gamma", 1.4)
        v["p_amb"] = self._labeled_entry(g_flow, "P ambient (psia)", 0.0)
        v["vel"] = self._labeled_entry(g_flow, "Velocity (ft/s)", 3022.0)
        v["isp_ideal"] = self._labeled_entry(g_flow, "Ideal Isp (lbf-s/lbm)",
                                             100.0)

        # --- Output streamlines + Print + Limiters ---
        g_sl = ttk.LabelFrame(right, text="Number of Output Streamlines")
        g_sl.grid(row=1, column=0, sticky="we", padx=4, pady=3)
        v["n_sl_i"] = self._grid_entry(g_sl, "Radial", 10, 0, 0, 8)
        v["n_sl_j"] = self._grid_entry(g_sl, "Axial", 50, 0, 2, 8)

        g_pr = ttk.LabelFrame(right, text="Print Option")
        g_pr.grid(row=2, column=0, sticky="we", padx=4, pady=3)
        v["print_mode"] = tk.StringVar(value="Full")
        ttk.Combobox(g_pr, textvariable=v["print_mode"], state="readonly",
                     width=12, values=("Summary", "Full")).grid(
            row=0, column=0, padx=4, pady=2)

        g_lim = ttk.LabelFrame(right, text="MOC Limiters")
        g_lim.grid(row=3, column=0, sticky="we", padx=4, pady=3)
        v["n_rrc_above_bd"] = self._labeled_entry(g_lim, "# of RRC above BD",
                                                  100)
        v["d_t_limit"] = self._labeled_entry(g_lim, "DTHETAB Max (deg)", 0.5)
        v["theta_bi"] = self._labeled_entry(g_lim, "THETAB Guess", 25.0)
        v["n_c"] = self._labeled_entry(g_lim, "# Starting Characteristics", 101)

        # --- output dir + buttons ---
        g_out = ttk.LabelFrame(right, text="Output")
        g_out.grid(row=4, column=0, sticky="we", padx=4, pady=3)
        g_out.columnconfigure(1, weight=1)
        v["out_dir"] = tk.StringVar(value=".")
        self._dir_picker(g_out, "Output folder", v["out_dir"], 0)

        btns = ttk.Frame(right)
        btns.grid(row=5, column=0, sticky="we", pady=6)
        ttk.Button(btns, text="Load .inp",
                   command=self._moc2d_load).pack(side="left", padx=3)
        ttk.Button(btns, text="Save .inp",
                   command=self._moc2d_save).pack(side="left", padx=3)
        b = ttk.Button(btns, text="Calculate MOC Grid",
                       command=self._run_moc2d)
        b.pack(side="right", padx=3)
        self._run_buttons.append(b)

        # --- plot ---
        pf, self._moc2d_draw = self._plot_area(right)
        pf.grid(row=6, column=0, sticky="nsew", padx=4, pady=4)
        right.rowconfigure(6, weight=1)
        right.columnconfigure(0, weight=1)

        self._moc2d_vars = v

    def _moc2d_collect(self):
        v = self._moc2d_vars
        inp = MOCInput()
        inp.axi = v["geom"].get() == "axi"
        inp.two_d = v["geom"].get() == "2d"
        t = v["type"].get()
        inp.perfect = t == "perfect"
        inp.min_length = t == "rao"
        inp.end_point = t == "endpoint"
        inp.cone = t == "cone"
        dp = v["dp"].get()
        inp.exit_mach = dp == "mach"
        inp.eps = dp == "eps"
        inp.length = dp == "length"
        inp.p_exit = dp == "pexit"
        inp.m_design = _f(v["dp_val"]["mach"].get(), 4.0)
        inp.eps_value = _f(v["dp_val"]["eps"].get(), 15.0)
        inp.length_value = _f(v["dp_val"]["length"].get(), 5.03258)
        inp.p_exit_value = _f(v["dp_val"]["pexit"].get(), 13.691)
        inp.x_e = _f(v["x_e"].get(), 4.308)
        inp.r_e = _f(v["r_e"].get(), 1.6)
        inp.cone_angle = _f(v["cone_angle"].get(), 15.0)
        inp.rwt_u = _f(v["rwt_u"].get(), 1.0)
        inp.rwt_d = _f(v["rwt_d"].get(), 1.0)
        inp.throat = bool(v["throat"].get())
        inp.pres_i = _f(v["pres_i"].get(), 1000.0)
        inp.temp_i = _f(v["temp_i"].get(), 530.0)
        inp.mol_wt_i = _f(v["mol_wt_i"].get(), 28.96)
        inp.gamma_i = _f(v["gamma_i"].get(), 1.4)
        inp.p_amb = _f(v["p_amb"].get(), 0.0)
        inp.vel = _f(v["vel"].get(), 3022.0)
        inp.isp_ideal = _f(v["isp_ideal"].get(), 100.0)
        inp.n_sl_i = _i(v["n_sl_i"].get(), 10)
        inp.n_sl_j = _i(v["n_sl_j"].get(), 50)
        inp.n_rrc_above_bd = _i(v["n_rrc_above_bd"].get(), 100)
        inp.d_t_limit = _f(v["d_t_limit"].get(), 0.5)
        inp.theta_bi = _f(v["theta_bi"].get(), 25.0)
        inp.n_c = _i(v["n_c"].get(), 101)
        inp.print_mode = 1 if v["print_mode"].get() == "Full" else 0
        return inp

    def _moc2d_apply(self, inp: MOCInput):
        v = self._moc2d_vars
        v["geom"].set("axi" if inp.axi else "2d")
        v["type"].set("perfect" if inp.perfect else "rao" if inp.min_length
                      else "endpoint" if inp.end_point else "cone")
        v["dp"].set("mach" if inp.exit_mach else "eps" if inp.eps
                    else "length" if inp.length else "pexit")
        v["dp_val"]["mach"].set(inp.m_design)
        v["dp_val"]["eps"].set(inp.eps_value)
        v["dp_val"]["length"].set(inp.length_value)
        v["dp_val"]["pexit"].set(inp.p_exit_value)
        for k in ("x_e", "r_e", "cone_angle", "rwt_u", "rwt_d", "pres_i",
                  "temp_i", "mol_wt_i", "gamma_i", "p_amb", "vel", "isp_ideal",
                  "n_sl_i", "n_sl_j", "n_rrc_above_bd", "d_t_limit",
                  "theta_bi", "n_c"):
            v[k].set(getattr(inp, k))
        v["throat"].set(inp.throat)

    def _moc2d_load(self):
        p = filedialog.askopenfilename(filetypes=[("MOC input", "*.inp"),
                                                  ("All", "*.*")])
        if p:
            try:
                self._moc2d_apply(read_moc2d_inp(p))
                self._status.set(f"Loaded {p}")
            except Exception as e:
                messagebox.showerror("Load failed", str(e))

    def _moc2d_save(self):
        p = filedialog.asksaveasfilename(defaultextension=".inp",
                                         filetypes=[("MOC input", "*.inp")])
        if p:
            try:
                write_moc2d_inp(self._moc2d_collect(), p)
                self._status.set(f"Saved {p}")
            except Exception as e:
                messagebox.showerror("Save failed", str(e))

    def _run_moc2d(self):
        inp = self._moc2d_collect()
        out = self._moc2d_vars["out_dir"].get().strip() or "."
        full = inp.print_mode == 1

        def work():
            calc = _build_calc(inp, full_output=full)
            result = calc.run()
            if not result.success:
                print("error:", result.error_message or "MOC solver failed")
                return 1, None
            moc2d_write_all(result, Path(out), full_output=full)
            import math
            print(f"Wrote results to {Path(out).resolve()}")
            print(f"  ThetaB(deg)      = {result.theta_b_ans*180/math.pi:.5g}")
            print(f"  last RRC         = {result.last_rrc}")
            print(f"  exit Mach (wall) = "
                  f"{result.grid.mach[0, result.last_rrc]:.5g}")
            return 0, result

        self._launch(work, "2D MOC", on_success=self._plot_moc2d)

    def _plot_moc2d(self, result):
        g = result.grid
        n = result.last_rrc
        xw = g.x[0, :n + 1]
        rw = g.r[0, :n + 1]

        def plot(ax):
            ax.plot(xw, rw, "-", color="#4f9dde", lw=1.6)
            ax.plot(xw, -rw, "-", color="#4f9dde", lw=1.6)
            ax.fill_between(xw, rw, -rw, color="#4f9dde", alpha=0.10)
            ax.axhline(0, color="#888", lw=0.6, ls="--")
            ax.set_xlabel("X / R*")
            ax.set_ylabel("R / R*")
            ax.set_title("Nozzle Contour")
            ax.set_aspect("equal", adjustable="datalim")

        self._moc2d_draw(plot)

    # ===================================================================
    #  STT tab
    # ===================================================================
    def _build_stt_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Streamline Trace  (STT2001)")

        top = ttk.Frame(tab)
        top.pack(side="top", fill="x", padx=4, pady=4)

        v = {}
        # File input group
        g_file = ttk.LabelFrame(top, text="File Input")
        g_file.grid(row=0, column=0, sticky="nwe", padx=4, pady=3)
        g_file.columnconfigure(1, weight=1)
        v["in_dir"] = tk.StringVar(value=".")
        self._dir_picker(g_file, "MOC input folder", v["in_dir"], 0)
        v["sl_filename"] = tk.StringVar(value="MOC_SL.plt")
        v["moc_grid_file"] = tk.StringVar(value="MOC_Grid.plt")
        v["moc_summary_filename"] = tk.StringVar(value="summary.out")
        v["friction_file"] = tk.StringVar(value="")
        v["file_prefix"] = tk.StringVar(value="case")
        for i, (lab, key) in enumerate([
                ("Streamline Data File", "sl_filename"),
                ("MOC Data File", "moc_grid_file"),
                ("MOC Summary File", "moc_summary_filename"),
                ("Friction Data File", "friction_file"),
                ("Output Files Prefix", "file_prefix")], start=1):
            ttk.Label(g_file, text=lab).grid(row=i, column=0, sticky="w",
                                             padx=4, pady=2)
            ttk.Entry(g_file, textvariable=v[key], width=24).grid(
                row=i, column=1, sticky="we", padx=4, pady=2)

        # Other inputs group
        g_other = ttk.LabelFrame(top, text="Other Inputs")
        g_other.grid(row=0, column=1, sticky="nwe", padx=4, pady=3)
        v["a_throat"] = self._labeled_entry(g_other, "Throat Area", 0.7854)
        v["p_ambient"] = self._labeled_entry(g_other, "Amb. Pressure (psia)",
                                             0.0)
        v["isp_ideal"] = self._labeled_entry(g_other, "Ideal Cfg Isp", 100.0)
        v["mass_flow"] = self._labeled_entry(g_other, "Mass Flow", 35.7)

        # SL flow-field centre sweep
        g_ctr = ttk.LabelFrame(top, text="Center of Known SL Flowfield")
        g_ctr.grid(row=0, column=2, sticky="nwe", padx=4, pady=3)
        ttk.Label(g_ctr, text="").grid(row=0, column=0)
        for j, h in enumerate(("XSL", "YSL", "RSL")):
            ttk.Label(g_ctr, text=h).grid(row=0, column=j + 1)
        for i, name in enumerate(("Start", "End", "Step"), start=1):
            ttk.Label(g_ctr, text=name).grid(row=i, column=0, sticky="w",
                                             padx=2)
            for j, ax in enumerate(("X", "Y", "R")):
                key = f"{ax}SL_{name.lower()}"
                dflt = 1.0 if (ax == "R") else 0.0
                sv = tk.StringVar(value=str(dflt))
                ttk.Entry(g_ctr, textvariable=sv, width=6).grid(
                    row=i, column=j + 1, padx=2, pady=1)
                v[key] = sv

        # Trimming slots
        g_trim = ttk.LabelFrame(tab, text="Streamline Trimming Definition")
        g_trim.pack(side="top", fill="x", padx=8, pady=4)
        headers = ["On", "Yc", "Zc", "Rc", "Start ang", "End ang",
                   "# SLs", "X Start", "X End", "Throat", "Surface"]
        for j, h in enumerate(headers):
            ttk.Label(g_trim, text=h, width=8, anchor="center").grid(
                row=0, column=j, padx=1)
        v["slots"] = []
        for r in range(1, 6):
            slot = {}
            slot["constraint"] = tk.BooleanVar(value=(r == 1))
            ttk.Checkbutton(g_trim, variable=slot["constraint"]).grid(
                row=r, column=0)
            for j, key in enumerate(["yc", "zc", "rc", "alpha", "omega",
                                     "nSL", "x_start", "x_end"], start=1):
                dflt = {"rc": 1.0, "omega": 360.0,
                        "nSL": 72}.get(key, 0.0) if r == 1 else 0.0
                sv = tk.StringVar(value=str(dflt))
                ttk.Entry(g_trim, textvariable=sv, width=8).grid(
                    row=r, column=j, padx=1, pady=1)
                slot[key] = sv
            slot["throat"] = tk.BooleanVar(value=(r == 1))
            ttk.Checkbutton(g_trim, variable=slot["throat"]).grid(
                row=r, column=9)
            slot["surface"] = tk.StringVar(value="inner")
            ttk.Combobox(g_trim, textvariable=slot["surface"], width=6,
                         state="readonly", values=("inner", "outer")).grid(
                row=r, column=10, padx=1)
            v["slots"].append(slot)

        # extra trim options
        g_xtra = ttk.Frame(g_trim)
        g_xtra.grid(row=6, column=0, columnspan=11, sticky="w", pady=3)
        v["max_length_check"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(g_xtra, text="Max X0", variable=v["max_length_check"]
                        ).pack(side="left", padx=2)
        v["max_length_c"] = tk.StringVar(value="0")
        ttk.Entry(g_xtra, textvariable=v["max_length_c"], width=8).pack(
            side="left", padx=2)
        ttk.Label(g_xtra, text="X Status").pack(side="left", padx=2)
        v["x_status"] = tk.StringVar(value="0")
        ttk.Entry(g_xtra, textvariable=v["x_status"], width=8).pack(
            side="left", padx=2)
        ttk.Label(g_xtra, text="Grid SF").pack(side="left", padx=2)
        v["grid_sf"] = tk.StringVar(value="1")
        ttk.Entry(g_xtra, textvariable=v["grid_sf"], width=6).pack(
            side="left", padx=2)
        v["contour_flag"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(g_xtra, text="Crop nozzle due to SL trimming",
                        variable=v["contour_flag"]).pack(side="left", padx=8)

        # outputs + run
        bottom = ttk.Frame(tab)
        bottom.pack(side="top", fill="both", expand=True, padx=8, pady=4)
        g_res = ttk.LabelFrame(bottom, text="Nozzle Parameter Output")
        g_res.pack(side="left", fill="y", padx=4)
        self._stt_out_vars = {}
        out_fields = [("Surface Area", "surface_area"),
                      ("Axial Projected Area", "projected_area"),
                      ("Throat Stream Thrust", "throat_thrust"),
                      ("P exit Force", "pressure_force"),
                      ("Friction Loss", "fric_loss"),
                      ("Exit Area", "a_exit"),
                      ("Calculated Isp", "isp_calc"),
                      ("Cfg", "cxx")]
        for i, (lab, key) in enumerate(out_fields):
            ttk.Label(g_res, text=lab).grid(row=i, column=0, sticky="w",
                                            padx=4, pady=1)
            sv = tk.StringVar(value="-")
            ttk.Entry(g_res, textvariable=sv, width=12, state="readonly").grid(
                row=i, column=1, padx=4, pady=1)
            self._stt_out_vars[key] = sv

        ctrl = ttk.Frame(bottom)
        ctrl.pack(side="right", fill="both", expand=True, padx=4)
        g_o = ttk.LabelFrame(ctrl, text="Output")
        g_o.pack(side="top", fill="x")
        g_o.columnconfigure(1, weight=1)
        v["out_dir"] = tk.StringVar(value=".")
        self._dir_picker(g_o, "Output folder", v["out_dir"], 0)
        bb = ttk.Frame(ctrl)
        bb.pack(side="top", fill="x", pady=6)
        ttk.Button(bb, text="Load .inp", command=self._stt_load).pack(
            side="left", padx=3)
        ttk.Button(bb, text="Save .inp", command=self._stt_save).pack(
            side="left", padx=3)
        b = ttk.Button(bb, text="Execute", command=self._run_stt)
        b.pack(side="right", padx=3)
        self._run_buttons.append(b)

        self._stt_vars = v

    def _stt_collect(self):
        v = self._stt_vars
        inp = STTInput()
        for k in ("sl_filename", "moc_grid_file", "moc_summary_filename",
                  "friction_file", "file_prefix"):
            setattr(inp, k, v[k].get().strip())
        inp.a_throat = _f(v["a_throat"].get())
        inp.p_ambient = _f(v["p_ambient"].get())
        inp.isp_ideal = _f(v["isp_ideal"].get(), 100.0)
        inp.mass_flow = _f(v["mass_flow"].get())
        for ax in ("X", "Y", "R", "Z"):
            for name in ("start", "end", "step"):
                key = f"{ax}SL_{name}"
                if key in v:
                    setattr(inp, f"{ax}SL_{name}", _f(v[key].get(),
                            1.0 if ax == "R" else 0.0))
        inp.max_length_check = bool(v["max_length_check"].get())
        inp.max_length_c = _f(v["max_length_c"].get())
        inp.x_status = _f(v["x_status"].get())
        inp.grid_sf = _i(v["grid_sf"].get(), 1)
        inp.contour_flag = bool(v["contour_flag"].get())
        inp.slots = []
        for s in v["slots"]:
            slot = ConstraintSlot()
            slot.constraint = bool(s["constraint"].get())
            slot.throat = bool(s["throat"].get())
            slot.surface = 0 if s["surface"].get() == "inner" else 1
            slot.yc = _f(s["yc"].get())
            slot.zc = _f(s["zc"].get())
            slot.rc = _f(s["rc"].get())
            slot.alpha = _f(s["alpha"].get())
            slot.omega = _f(s["omega"].get())
            slot.nSL = _i(s["nSL"].get())
            slot.x_start = _f(s["x_start"].get())
            slot.x_end = _f(s["x_end"].get())
            inp.slots.append(slot)
        return inp

    def _stt_apply(self, inp: STTInput):
        v = self._stt_vars
        for k in ("sl_filename", "moc_grid_file", "moc_summary_filename",
                  "friction_file", "file_prefix"):
            v[k].set(getattr(inp, k))
        v["a_throat"].set(inp.a_throat)
        v["p_ambient"].set(inp.p_ambient)
        v["isp_ideal"].set(inp.isp_ideal)
        v["mass_flow"].set(inp.mass_flow)
        for ax in ("X", "Y", "R"):
            for name in ("start", "end", "step"):
                key = f"{ax}SL_{name}"
                if key in v:
                    v[key].set(getattr(inp, f"{ax}SL_{name}"))
        v["max_length_check"].set(inp.max_length_check)
        v["max_length_c"].set(inp.max_length_c)
        v["x_status"].set(inp.x_status)
        v["grid_sf"].set(inp.grid_sf)
        v["contour_flag"].set(inp.contour_flag)
        for s, slot in zip(v["slots"], inp.slots):
            s["constraint"].set(slot.constraint)
            s["throat"].set(slot.throat)
            s["surface"].set("inner" if slot.surface == 0 else "outer")
            s["yc"].set(slot.yc)
            s["zc"].set(slot.zc)
            s["rc"].set(slot.rc)
            s["alpha"].set(slot.alpha)
            s["omega"].set(slot.omega)
            s["nSL"].set(slot.nSL)
            s["x_start"].set(slot.x_start)
            s["x_end"].set(slot.x_end)

    def _stt_load(self):
        p = filedialog.askopenfilename(filetypes=[("STT input", "*.inp"),
                                                  ("All", "*.*")])
        if p:
            try:
                self._stt_apply(read_stt_inp(p))
                self._stt_vars["in_dir"].set(str(Path(p).parent))
                self._status.set(f"Loaded {p}")
            except Exception as e:
                messagebox.showerror("Load failed", str(e))

    def _stt_save(self):
        p = filedialog.asksaveasfilename(defaultextension=".inp",
                                         filetypes=[("STT input", "*.inp")])
        if p:
            try:
                write_stt_inp(self._stt_collect(), p)
                self._status.set(f"Saved {p}")
            except Exception as e:
                messagebox.showerror("Save failed", str(e))

    def _run_stt(self):
        inp = self._stt_collect()
        in_dir = self._stt_vars["in_dir"].get().strip() or "."
        out = self._stt_vars["out_dir"].get().strip() or "."

        def work():
            result = STTSolver(in_dir, inp).run(Path(out))
            if not result.success:
                print("error:", result.error_message or "STT solver failed")
                return 1, None
            print(f"Wrote results to {Path(out).resolve()}")
            for lab, key in [("Surface area", "surface_area"),
                             ("Pressure force", "pressure_force"),
                             ("Throat thrust", "throat_thrust"),
                             ("Isp calc", "isp_calc"), ("Cfg", "cxx")]:
                print(f"  {lab:14}= {getattr(result, key):.5g}")
            return 0, result

        self._launch(work, "Streamline Trace", on_success=self._stt_show)

    def _stt_show(self, result):
        for key, sv in self._stt_out_vars.items():
            try:
                sv.set(f"{getattr(result, key):.6g}")
            except Exception:
                sv.set("-")

    # ===================================================================
    #  3D MOC tab
    # ===================================================================
    def _build_moc3d_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="3D MOC  (3D_MOC)")
        v = {}

        g_io = ttk.LabelFrame(tab, text="File Input / Output")
        g_io.grid(row=0, column=0, sticky="nwe", padx=6, pady=4)
        g_io.columnconfigure(1, weight=1)
        v["geo"] = tk.StringVar(value="")
        self._file_picker(g_io, "Geometry Input File",
                          v["geo"], [("Wall contour", "*.geo"),
                                     ("All", "*.*")], 0)
        v["out_dir"] = tk.StringVar(value=".")
        self._dir_picker(g_io, "Output folder", v["out_dir"], 1)

        g_ip = ttk.LabelFrame(tab, text="Initial Plane Properties")
        g_ip.grid(row=0, column=1, sticky="nwe", padx=6, pady=4)
        v["p0"] = self._labeled_entry(g_ip, "Pressure (psia)", 1000.0)
        v["t0"] = self._labeled_entry(g_ip, "Temperature (R)", 530.0)
        v["mach0"] = self._labeled_entry(g_ip, "Mach Number", 1.1)
        v["mol_wt0"] = self._labeled_entry(g_ip, "Mol. Wt.", 28.96)
        v["gamma0"] = self._labeled_entry(g_ip, "Gamma", 1.4)
        v["theta0"] = self._labeled_entry(g_ip, "Theta (deg)", 0.0)
        v["psi0"] = self._labeled_entry(g_ip, "Psi (deg)", 0.0)

        g_grid = ttk.LabelFrame(tab, text="Grid Setup (Cylindrical)")
        g_grid.grid(row=1, column=0, sticky="nwe", padx=6, pady=4)
        v["n_div"] = self._labeled_entry(g_grid, "Radial Divisions", 36)

        g_pr = ttk.LabelFrame(tab, text="Print Output Parameters")
        g_pr.grid(row=1, column=1, sticky="nwe", padx=6, pady=4)
        v["x_step"] = self._labeled_entry(g_pr, "Every N point(s) in X", 1)
        v["y_step"] = self._labeled_entry(g_pr, "Every N point(s) in Y", 1)
        v["z_step"] = self._labeled_entry(g_pr, "Every N point(s) in Z", 1)
        v["step_step"] = self._labeled_entry(g_pr, "Every N step number", 999)

        g_sf = ttk.LabelFrame(tab, text="Surface Fit")
        g_sf.grid(row=2, column=0, sticky="nwe", padx=6, pady=4)
        v["surface_fit"] = tk.StringVar(value="All Point Spline")
        ttk.Combobox(g_sf, textvariable=v["surface_fit"], state="readonly",
                     width=18,
                     values=("All Point Spline", "9 Point Spline")).pack(
            padx=6, pady=4)

        bb = ttk.Frame(tab)
        bb.grid(row=2, column=1, sticky="e", padx=6, pady=4)
        b = ttk.Button(bb, text="Calculate Nozzle", command=self._run_moc3d)
        b.pack(side="right")
        self._run_buttons.append(b)

        ttk.Label(tab, foreground="#888",
                  text="Tip: set Z stride = 1 to write every plane (matches "
                       "the sample). The full cone10 march takes ~1 minute."
                  ).grid(row=3, column=0, columnspan=2, sticky="w", padx=8)

        pf, self._moc3d_draw = self._plot_area(tab)
        pf.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=6, pady=4)
        tab.rowconfigure(4, weight=1)
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        self._moc3d_vars = v

    def _run_moc3d(self):
        v = self._moc3d_vars
        geo = v["geo"].get().strip()
        out = v["out_dir"].get().strip() or "."
        if not geo:
            messagebox.showwarning("Missing input", "Choose a .geo file.")
            return

        def work():
            cfg = GeoConfig(
                p0=_f(v["p0"].get(), 1000.0), t0=_f(v["t0"].get(), 530.0),
                mach0=_f(v["mach0"].get(), 1.1),
                mol_wt0=_f(v["mol_wt0"].get(), 28.96),
                gamma0=_f(v["gamma0"].get(), 1.4),
                theta0=_f(v["theta0"].get(), 0.0),
                psi0=_f(v["psi0"].get(), 0.0),
                n_div=_i(v["n_div"].get(), 36),
                x_output_step=_i(v["x_step"].get(), 1),
                y_output_step=_i(v["y_step"].get(), 1),
                z_output_step=_i(v["z_step"].get(), 1),
                step_step=_i(v["step_step"].get(), 999),
                surface_fit=v["surface_fit"].get(),
            )
            read_geo(geo, cfg)
            grid = MOC3DGrid(cfg, Path(out))
            grid.set_initial_properties()
            result = grid.calc_nozzle()
            if not result.success:
                print("error:", result.error_message or "solver failed")
                return 1, None
            print(f"Wrote results to {Path(out).resolve()}")
            print(f"  Total field points : {result.n_pts}")
            print(f"  Axial planes       : {result.n_z}")
            print(f"  Angular divisions  : {result.n_div}")
            return 0, (result, cfg)

        self._launch(work, "3D MOC", on_success=self._plot_moc3d)

    def _plot_moc3d(self, payload):
        result, cfg = payload

        def plot(ax):
            ax.plot(cfg.z, cfg.r, "-", color="#4f9dde", lw=1.6)
            ax.plot(cfg.z, [-x for x in cfg.r], "-", color="#4f9dde", lw=1.6)
            ax.axhline(0, color="#888", lw=0.6, ls="--")
            ax.set_xlabel("Z (in)")
            ax.set_ylabel("R (in)")
            ax.set_title("Wall Contour (input geometry)")
            ax.set_aspect("equal", adjustable="datalim")

        self._moc3d_draw(plot)

    # ===================================================================
    #  run machinery
    # ===================================================================
    def _launch(self, work_fn, label, on_success=None):
        if self._busy:
            messagebox.showinfo("Busy", "A run is already in progress.")
            return
        self._busy = True
        for b in self._run_buttons:
            b.state(["disabled"])
        self._status.set(f"Running {label}...")
        self._append_log(f"\n=== {label} ===\n")

        def worker():
            writer = _QueueWriter(self._log_q)
            rc, result = 1, None
            try:
                with redirect_stdout(writer), redirect_stderr(writer):
                    rc, result = work_fn()
            except SystemExit as e:
                self._log_q.put(f"\n[stopped] {e}\n")
                rc = e.code if isinstance(e.code, int) else 1
            except Exception:
                self._log_q.put("\n[error]\n" + traceback.format_exc() + "\n")
                rc = 1
            self._log_q.put(("__DONE__", label, rc, result, on_success))

        threading.Thread(target=worker, daemon=True).start()

    def _drain_log(self):
        try:
            while True:
                item = self._log_q.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "__DONE__":
                    _, label, rc, result, on_success = item
                    self._finish(label, rc, result, on_success)
                else:
                    self._append_log(item)
        except queue.Empty:
            pass
        self.after(80, self._drain_log)

    def _finish(self, label, rc, result, on_success):
        self._busy = False
        for b in self._run_buttons:
            b.state(["!disabled"])
        if rc == 0:
            self._status.set(f"{label} finished successfully.")
            if on_success and result is not None:
                try:
                    on_success(result)
                except Exception:
                    self._append_log("\n[plot error]\n"
                                     + traceback.format_exc() + "\n")
        else:
            self._status.set(f"{label} failed (exit {rc}). See log.")

    def _append_log(self, text):
        self._log.configure(state="normal")
        self._log.insert("end", text)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _clear_log(self):
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")


def main():
    app = PynozzleGUI()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
