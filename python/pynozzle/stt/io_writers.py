"""STT2001 output writers — direct ports of the file-writing routines
in ``STT2001Dlg.cpp``.

The C++ writes a number of files for each run:
 * ``<prefix>_ThroatSL.out``      — ICEM CFD bulk-data dump of throat SLs
 * ``<prefix>_ThroatSummary.out`` — table summarising throat-plane SLs
 * ``<prefix>_TrimSL.out``        — ICEM CFD dump of trimmed SLs
 * ``<prefix>_trimmed_P3D.xyz``   — Plot3D grid of the trimmed nozzle
 * ``<prefix>_trimmed_P3D.dat``   — Plot3D pressure data
 * ``<prefix>_end_P3D.xyz``       — Plot3D of the SL end-points
 * ``<prefix>.plt``               — Tecplot of the trimmed nozzle
 * ``<prefix>_Engine.plt``        — Tecplot with full-engine revolution
 * ``<prefix>_STT_summary.out``   — human-readable performance summary
 * ``<prefix>_AvsX.out``          — running totals vs axial X
 * ``<prefix>_AvsSL.out``         — running totals vs streamline index
 * ``<prefix>_cl.dat``            — centerline geometry
 * ``<prefix>_all_runs.dat``      — sweep summary
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..common.constants import PI, DEG_PER_RAD

if TYPE_CHECKING:
    from .solver import STTState, STTResult
    from .inp import STTInput
    from .loaders import MOCSummaryData


# ---------------------------------------------------------------------------
def _fmt(x: float) -> str:
    """Match the default C++ ostream output (up to 6 sig figs)."""
    if x != x:  # NaN -- the C++ would have read 0 from a missing field
        return "0"
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e6 or ax < 1e-4:
        s = f"{x:.5e}"
        mant, exp = s.split("e")
        mant = mant.rstrip("0").rstrip(".")
        return f"{mant}e{int(exp):+03d}"
    return f"{x:.6g}"


# ---------------------------------------------------------------------------
def write_throat_sls(
    state: "STTState", out_dir: Path, prefix: str, level_start: int = 200,
) -> None:
    """Direct port of ``CSTT2001Dlg::OutputThroatSLsToFile``."""
    bulk_path = out_dir / f"{prefix}_ThroatSL.out"
    sum_path = out_dir / f"{prefix}_ThroatSummary.out"

    with open(bulk_path, "w") as f:
        f.write("REMARK/ streamline points data file created by STT2001\n")
        f.write("FORMAT/3F10.4\n")
        level = level_start - 1
        dist = 1.0
        for n in range(state.n_new_sls):
            if n > 0:
                dist = math.sqrt((state.yt[n, 0] - state.yt[n - 1, 0]) ** 2
                                 + (state.zt[n, 0] - state.zt[n - 1, 0]) ** 2)
            if dist > 0.01:
                level += 1
                f.write(f"LEVEL/{level}\n")
                f.write(f"POINT/{int(state.new_sl_pts[n])}\n")
                last_j = int(state.new_sl_pts[n])
                for j in range(last_j + 1):
                    f.write(f"{state.xt[n, j]:10.4f}{state.yt[n, j]:10.4f}"
                            f"{state.zt[n, j]:10.4f}\n")
        # first-point bundle
        level += 1
        f.write(f"LEVEL/{level}\n")
        f.write(f"POINT/{state.n_new_sls}\n")
        for n in range(state.n_new_sls):
            f.write(f"{state.xt[n, 0]:10.4f}{state.yt[n, 0]:10.4f}"
                    f"{state.zt[n, 0]:10.4f}\n")
        # last-point bundle
        level += 1
        f.write(f"LEVEL/{level}\n")
        f.write(f"POINT/{state.n_new_sls}\n")
        for n in range(state.n_new_sls):
            j = int(state.new_sl_pts[n])
            f.write(f"{state.xt[n, j]:10.4f}{state.yt[n, j]:10.4f}"
                    f"{state.zt[n, j]:10.4f}\n")

    # Throat-plane summary (one line per SL).
    # NOTE: in the C++ this is written through the same fstream object as
    # the bulk data above, so it inherits the ``fixed`` flag and
    # ``precision(4)`` set there -- hence 4-decimal fixed formatting.
    with open(sum_path, "w") as f:
        f.write(f"# \t  SL used(wall = {state.sl.n_sl}) \t XT \t YT \t ZT "
                "\t Pres \t Ideal end\n")
        for n in range(state.n_new_sls):
            f.write(f"{n}\t{int(state.nt[n])}\t"
                    f"{state.xt[n, 0]:.4f}\t{state.yt[n, 0]:.4f}\t"
                    f"{state.zt[n, 0]:.4f}\t{state.pt[n, 0]:.4f}\t"
                    f"{state.xt_end[n]:.4f}\n")


# ---------------------------------------------------------------------------
def write_trimmed_sls(
    state: "STTState", inp: "STTInput", out_dir: Path, prefix: str,
    level_start: int = 300,
) -> None:
    """Direct port of ``CSTT2001Dlg::OutputTrimmedSLsToFile``.

    Writes the trimmed SLs in ICEM CFD BULKIN format plus Plot3D and
    Tecplot files. Note (matching the C++): unlike the throat-SL output,
    this one uses ``d1 = 2`` precision, **omits** the ``FORMAT/3F10.4``
    line (it's commented out in the source), and starts levels at 300.
    """
    n_grid_x = state.n_param_grid_x
    d1 = 2  # precision used by the C++ here (the throat file uses 4)

    # --- ICEM BULKIN -------------------------------------------------
    bulk_path = out_dir / f"{prefix}_TrimSL.out"
    with open(bulk_path, "w") as f:
        f.write("REMARK/ streamline points data file created by STT2001\n")
        # NB: the C++ has the "FORMAT/3F10.4" line commented out here.
        level = level_start - 1
        dist = 1.0
        for n in range(state.n_new_sls):
            if n > 0:
                dist = math.sqrt(
                    (state.y_grid[n, 0] - state.y_grid[n - 1, 0]) ** 2
                    + (state.z_grid[n, 0] - state.z_grid[n - 1, 0]) ** 2)
            else:
                dist = 1.0
            if dist > 0.01:
                level += 1
                f.write(f"LEVEL/{level}\n")
                # determine number of unique points
                k = n_grid_x - 1
                for j in range(n_grid_x - 1):
                    if state.x_grid[n, j] == state.x_grid[n, j + 1]:
                        k = j
                        break
                f.write(f"POINT/{k + 1}\n")
                for j in range(k + 1):
                    f.write(f"{state.x_grid[n, j]:10.{d1}f}"
                            f"{state.y_grid[n, j]:10.{d1}f}"
                            f"{state.z_grid[n, j]:10.{d1}f}\n")
        # first-point bundle
        level += 1
        f.write(f"LEVEL/{level}\n")
        f.write(f"POINT/{state.n_new_sls}\n")
        for n in range(state.n_new_sls):
            f.write(f"{state.x_grid[n, 0]:10.{d1}f}"
                    f"{state.y_grid[n, 0]:10.{d1}f}"
                    f"{state.z_grid[n, 0]:10.{d1}f}\n")
        # last-point bundle
        level += 1
        f.write(f"LEVEL/{level}\n")
        f.write(f"POINT/{state.n_new_sls}\n")
        for n in range(state.n_new_sls):
            j = n_grid_x - 1
            f.write(f"{state.x_grid[n, j]:10.{d1}f}"
                    f"{state.y_grid[n, j]:10.{d1}f}"
                    f"{state.z_grid[n, j]:10.{d1}f}\n")

    # --- Plot3D xyz/dat ----------------------------------------------
    grid_path = out_dir / f"{prefix}_trimmed_P3D.xyz"
    data_path = out_dir / f"{prefix}_trimmed_P3D.dat"
    end_path = out_dir / f"{prefix}_end_P3D.xyz"
    with open(grid_path, "w") as gf, open(data_path, "w") as df, \
            open(end_path, "w") as ef:
        gf.write(f"{n_grid_x} {state.n_new_sls + 1}  1\n")
        df.write(f"{n_grid_x} {state.n_new_sls + 1}  1  1\n")
        ef.write(f"{state.n_new_sls + 1} 1 1\n")

        # X with end file. NOTE: the C++ uses ``if (k++ > 8)`` which is a
        # *post*-increment -- it tests the OLD value of k, then increments.
        # That yields 10 values per row, not 9. We replicate it exactly.
        for n in range(state.n_new_sls + 1):
            m = 0 if n == state.n_new_sls else n
            k = 0
            for j in range(n_grid_x):
                v = state.x_grid[m, j]
                if abs(v) > 0.00001:
                    gf.write(f"{v:10.{d1}f}")
                else:
                    gf.write(f"{0.0:10.{d1}f}")
                df.write(f"{state.p_grid[m, j]:10.{d1}f}")
                cond = k > 8
                k += 1
                if cond:
                    k = 0
                    gf.write("\n")
                    df.write("\n")
            ef.write(f"{state.xt_end[m]:10.{d1}f} ")
            gf.write("\n")
            df.write("\n")
        ef.write("\n")
        # Y
        for n in range(state.n_new_sls + 1):
            m = 0 if n == state.n_new_sls else n
            k = 0
            for j in range(n_grid_x):
                v = state.y_grid[m, j]
                if abs(v) > 0.00001:
                    gf.write(f"{v:10.{d1}f}")
                else:
                    gf.write(f"{0.0:10.{d1}f}")
                cond = k > 8
                k += 1
                if cond:
                    k = 0
                    gf.write("\n")
            ef.write(f"{state.yt_end[m]:10.{d1}f} ")
            gf.write("\n")
        ef.write("\n")
        # Z
        for n in range(state.n_new_sls + 1):
            m = 0 if n == state.n_new_sls else n
            k = 0
            for j in range(n_grid_x):
                v = state.z_grid[m, j]
                if abs(v) > 0.00001:
                    gf.write(f"{v:10.{d1}f}")
                else:
                    gf.write(f"{0.0:10.{d1}f}")
                cond = k > 8
                k += 1
                if cond:
                    k = 0
                    gf.write("\n")
            ef.write(f"{state.zt_end[m]:10.{d1}f} ")
            gf.write("\n")

    # --- Tecplot .plt -------------------------------------------------
    # NOTE: written through the same stream object as the BULKIN file in
    # the C++, so it inherits ``fixed`` + ``precision(2)``.
    tp_path = out_dir / f"{prefix}.plt"
    with open(tp_path, "w") as f:
        f.write('VARIABLES = "X(in)","Y(in)","Z(in)","Pressure"\n')
        f.write('TITLE = "STT2001 Ouput"\n')
        f.write('text  x=5  y=93  t="Trimmed Nozzle Data"\n')
        f.write(f'zone t="All data" I = {state.n_new_sls + 1}'
                f' J ={n_grid_x} K = 1\n')
        for j in range(n_grid_x):
            for n in range(state.n_new_sls + 1):
                m = 0 if n == state.n_new_sls else n
                f.write(f"{state.x_grid[m, j]:.{d1}f}\t{state.y_grid[m, j]:.{d1}f}\t"
                        f"{state.z_grid[m, j]:.{d1}f}\t{state.p_grid[m, j]:.{d1}f}\n")

    # --- Tecplot _Engine.plt ------------------------------------------
    # The C++ loops ``for i in range(n_rev)`` -- when n_rev is 0 (no
    # repeated revolution) the file contains only the header lines.
    engine_path = out_dir / f"{prefix}_Engine.plt"
    n_rev = inp.sym.n_rev
    y_sim = inp.sym.Y_sim
    with open(engine_path, "w") as f:
        f.write('VARIABLES = "X(in)","Y(in)","Z(in)","Pressure"\n')
        f.write('TITLE = "STT2001 Ouput"\n')
        f.write('text  x=5  y=93  t="Trimmed Nozzle Data"\n')
        for i in range(n_rev):
            f.write(f'zone t="All data" I = {state.n_new_sls + 1}'
                    f' J ={n_grid_x} K = 1\n')
            for j in range(n_grid_x):
                for n in range(state.n_new_sls + 1):
                    m = 0 if n == state.n_new_sls else n
                    S = math.sqrt(state.z_grid[m, j] ** 2
                                  + (state.y_grid[m, j] - y_sim) ** 2)
                    if (state.y_grid[m, j] - y_sim) != 0.0:
                        thetap = math.atan(state.z_grid[m, j]
                                           / (state.y_grid[m, j] - y_sim))
                    else:
                        thetap = math.pi / 2
                    thetat = thetap + i * 2 * PI / n_rev
                    Y = S * math.cos(thetat)
                    Z = S * math.sin(thetat)
                    f.write(f"{state.x_grid[m, j]:.{d1}f}\t{Y:.{d1}f}\t{Z:.{d1}f}\t"
                            f"{state.p_grid[m, j]:.{d1}f}\n")


# ---------------------------------------------------------------------------
def write_stt_summary(
    state: "STTState", inp: "STTInput", moc_data: "MOCSummaryData",
    result: "STTResult", out_dir: Path,
) -> None:
    """Direct port of the summary-writing block at the end of
    ``CSTT2001Dlg::GetPerformanceDataFromMOCSummaryFile``.
    """
    prefix = inp.file_prefix
    a_exit = result.a_exit
    cxx = result.cxx
    isp_calc = result.isp_calc
    p_a_loss = result.pa_loss
    fric_loss = result.fric_loss
    total_loss = fric_loss + p_a_loss
    fric_ratio = fric_loss / (inp.mass_flow * inp.isp_ideal) if (
        inp.mass_flow and inp.isp_ideal) else 0.0
    pa_ratio = p_a_loss / (inp.mass_flow * inp.isp_ideal) if (
        inp.mass_flow and inp.isp_ideal) else 0.0

    path = out_dir / f"{prefix}_STT_summary.out"
    with open(path, "w") as f:
        f.write("This is the summary file for STT2001\n\n")
        f.write("MOC DATA for a full nozzle\n")
        f.write(f"Throat Radius (in):\t{_fmt(moc_data.r_star_moc)}\n")
        f.write(f"Mass Flow (lbm/s):\t{_fmt(moc_data.mdot_moc)}\n")
        f.write(f"Area Ratio:\t{_fmt(moc_data.eps_moc)}\n")
        f.write(f"Exit Area (in2):\t{_fmt(moc_data.a_exit_moc)}\n")
        f.write(f"Surface Area (in2):\t{_fmt(moc_data.a_surf_moc)}\n")
        f.write(f"2-D Stream Thrust @ throat(lbf):\t{_fmt(moc_data.thrust_1_moc)}\n")
        f.write(f"Stream Thrust @ Exit (lbf):\t{_fmt(moc_data.s_exit_moc)}\n")
        f.write(f"Ambient Pressure (psia):\t{_fmt(moc_data.p_amb_moc)}\n")
        f.write(f"Gross Thrust @ exit(lbf):\t{_fmt(moc_data.f_exit_moc)}\n")
        f.write(f"Gross ISP at Exit (lbf-sec/lbm):\t{_fmt(moc_data.isp_2d_moc)}\n")
        f.write("\nSTT2001 Calculated Parameters\n")
        f.write(f"Nozzle Surface Area (in2):\t{_fmt(state.m_surface_area)}\n")
        f.write(f"Nozzle Throat Area (in2):\t{_fmt(inp.a_throat)}\n")
        f.write(f"Nozzle Wall Axial Projected Area (in2):\t{_fmt(state.m_projected_area)}\n")
        f.write(f"Nozzle Exit Area (in2):\t{_fmt(a_exit)}\n")
        f.write(f"Nozzle Area Ratio:\t{_fmt(a_exit/inp.a_throat if inp.a_throat else 0)}\n")
        f.write(f"Mass Flow (lbm/s):\t{_fmt(inp.mass_flow)}\n")
        f.write(f"Throat Thrust(lbf):\t{_fmt(result.throat_thrust)}\n")
        f.write(f"Pressure Force (lbf):\t{_fmt(state.m_force)}\n")
        f.write(f"Exit Stream Thrust (lbf):\t{_fmt(result.throat_thrust + state.m_force)}\n")
        f.write(f"Pa Ae Loss (lbf):\t{_fmt(-p_a_loss)}\n")
        f.write(f"Exit Gross Thrust (lbf):\t{_fmt(result.throat_thrust + state.m_force - p_a_loss)}\n")
        f.write(f"Friction Loss (lbf):\t{_fmt(-fric_loss)}\n")
        f.write(
            f"Exit Gross Thrust w/ Friction (lbf):\t"
            f"{_fmt(result.throat_thrust + state.m_force - fric_loss - p_a_loss)}\n\n")
        f.write(f"Calculated Isp (lbf-s/lbm)\t{_fmt(isp_calc)}\n")
        f.write(f"Ideal Isp (lbf-s/lbm)\t{_fmt(inp.isp_ideal)}\n")
        f.write(f"Cfg\t{_fmt(cxx)}\n")
        f.write("Losses Summary\tCfg Loss\t(% of Total Loss)\n")
        if total_loss > 0.0:
            f.write(f"Friction Loss\t{_fmt(fric_ratio)}\t"
                    f"{_fmt(fric_loss * 100 / total_loss)}\n")
            f.write(f"Exit Pressure Loss\t{_fmt(pa_ratio)}\t"
                    f"{_fmt(p_a_loss * 100 / total_loss)}\n")
        else:
            f.write(f"Friction Loss\t{_fmt(fric_ratio)}\t0\n")
            f.write(f"Exit Pressure Loss\t{_fmt(pa_ratio)}\t0\n")


# ---------------------------------------------------------------------------
def write_centerline_plot(
    state: "STTState", out_dir: Path, prefix: str,
) -> None:
    """Direct port of ``CSTT2001Dlg::OutputCenterlinePlot``.

    Writes ``<prefix>_cl.dat`` with the max-Y and min-Y streamline
    contours and their local wall angles. The C++ calls this with
    ``xgrid[m_NAtMaxY]`` / ``ygrid[m_NAtMaxY]`` and
    ``xgrid[m_NAtMinY]`` / ``ygrid[m_NAtMinY]``.
    """
    n = state.n_param_grid_x
    x1 = state.x_grid[state.n_at_max_y]
    y1 = state.y_grid[state.n_at_max_y]
    x2 = state.x_grid[state.n_at_min_y]
    y2 = state.y_grid[state.n_at_min_y]
    path = out_dir / f"{prefix}_cl.dat"
    with open(path, "w") as f:
        f.write("Xc (in) \t Yc (in) \t Tc (deg) \t\t Xb (in)\tYb (in)\tTb (deg)\n")
        theta1 = theta2 = 0.0
        for i in range(n):
            if i > 0:
                if x1[i] != x1[i - 1]:
                    theta1 = math.atan((y1[i] - y1[i - 1])
                                       / (x1[i] - x1[i - 1])) * DEG_PER_RAD
                if x2[i] != x2[i - 1]:
                    theta2 = math.atan((y2[i] - y2[i - 1])
                                       / (x2[i] - x2[i - 1])) * DEG_PER_RAD
            else:
                theta1 = theta2 = 0.0
            f.write(f"{_fmt(x1[i])}\t{_fmt(y1[i])}\t{_fmt(theta1)}\t\t"
                    f"{_fmt(x2[i])}\t{_fmt(y2[i])}\t{_fmt(theta2)}\n")


# ---------------------------------------------------------------------------
_ALL_RUNS_HEADER = (
    "Prefix \t ZSL \t XSL \t YSL \t RSL \t ISP \t Cfg \t Surface Area (in2) \t"
    "Throat Area (in2)\t Axial Projected Area (in2)\t Exit Area (in2)\t"
    "Area Ratio\t Area Ratio(MOC)\t Mass Flow (lbm/s)\t"
    "Throat Momentum (lbf)\t Pressure Force (lbf)\t Pa Ae Loss (lbf)\t"
    "Friction Loss (lbf) \t Max Length (in) \t Min Y(in) \t Max Y(in)\t"
    "X @ Min Y(in) \t X @ Max Y(in) \t X Status (in) \t XStatus Ymin (in)\t"
    "XStatus Theta @ Min Y (deg)\n"
)


def write_all_runs_header(out_dir: Path, prefix: str):
    """Open ``<prefix>_all_runs.dat`` and write the header; return the handle."""
    f = open(out_dir / f"{prefix}_all_runs.dat", "w")
    f.write(_ALL_RUNS_HEADER)
    return f


def write_all_runs_row(
    f, inp: "STTInput", state: "STTState", result: "STTResult",
    z_sl: float, x_sl: float, y_sl: float, r_sl: float,
    y_x_status: float, theta_x_status: float,
) -> None:
    """Write one sweep row to the already-open all_runs file."""
    isp = result.isp_calc
    cfg = result.cxx
    f.write(
        f"{inp.sl_filename}\t{_fmt(z_sl)}\t{_fmt(x_sl)}\t{_fmt(y_sl)}\t"
        f"{_fmt(r_sl)}\t{_fmt(isp)}\t{_fmt(cfg)}\t{_fmt(state.m_surface_area)}\t"
        f"{_fmt(inp.a_throat)}\t{_fmt(state.m_projected_area)}\t{_fmt(result.a_exit)}\t"
        f"{_fmt(result.a_exit/inp.a_throat if inp.a_throat else 0)}\t"
        f"{_fmt(result.eps_moc)}\t{_fmt(inp.mass_flow)}\t"
        f"{_fmt(result.throat_thrust)}\t{_fmt(state.m_force)}\t{_fmt(-result.pa_loss)}\t"
        f"{_fmt(-result.fric_loss)}\t{_fmt(state.max_length)}\t{_fmt(state.min_y)}\t"
        f"{_fmt(state.max_y)}\t{_fmt(state.x_at_min_y)}\t{_fmt(state.x_at_max_y)}\t"
        f"{_fmt(inp.x_status)}\t{_fmt(y_x_status)}\t{_fmt(theta_x_status)}\n"
    )
