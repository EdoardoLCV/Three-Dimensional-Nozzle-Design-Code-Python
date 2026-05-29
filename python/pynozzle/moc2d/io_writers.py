"""Output-file writers for :mod:`pynozzle.moc2d`.

These are direct ports of the routines in ``MOC_GridCalc_BDE_IO.cpp``.
Each function writes a file with the same name and structure as the
original C++ code, so the sample outputs in
``outputs_M3.5Perf`` / ``outputs_M4RAO`` / ``outputs_cone10`` can be
diffed directly for validation.

The C++ code used the default ``operator<<`` for doubles, which prints
at most 6 significant digits with no trailing zeros (e.g. ``12.5363`` not
``12.53630000``). We reproduce that with ``_fmt`` below.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TextIO

import numpy as np

from ..common.constants import (
    AXI, CONE, DEG_PER_RAD, ENDPOINT, EPS, EXITMACH, EXITPRESSURE,
    FIXEDEND, GASCON, GRAV, NOZZLELENGTH, PERFECT, PI, RAO, TWOD,
)
from ..common.thermo import isentropic_p_t_rho
from .solver import MOCResult


# ---------------------------------------------------------------------------
def _fmt(x: float) -> str:
    """Format a double the way C++ ``ostream``'s default operator does:
    up to 6 significant digits, no scientific until very small / large."""
    if x == 0.0:
        return "0"
    abs_x = abs(x)
    if abs_x >= 1e6 or abs_x < 1e-4:
        # use scientific
        s = f"{x:.5e}"
        # tidy: drop trailing zeros in mantissa
        mant, exp = s.split("e")
        mant = mant.rstrip("0").rstrip(".")
        return f"{mant}e{int(exp):+03d}"
    s = f"{x:.6g}"
    return s


def _row(*vals) -> str:
    return "\t".join(_fmt(v) if isinstance(v, float) else str(v) for v in vals)


# ---------------------------------------------------------------------------
def write_summary(result: MOCResult, path: str = "summary.out") -> None:
    """Direct port of ``MOC_GridCalc::OutputSummaryFile``.

    Writes a human-readable summary of the MOC solution including the
    initial data line, kernel mass flow / thrust, the line DE, the
    calculated wall contour, the exit-plane profile, and the
    performance scalars.
    """
    g = result.grid
    assert g is not None

    last_rrc = result.last_rrc
    n_type = result.nozzle_type
    geom = result.nozzle_geom
    design_param = result.design_param
    design_value = result.design_value
    mol_wt = result.mol_wt
    gamma_i = result.gamma_i
    p_total = result.p_total
    t_total = result.t_total
    p_ambient = result.p_ambient
    isp_ideal = result.isp_ideal

    with open(path, "w") as f:
        f.write("Summary output file for CONGO-2D\n\n")
        f.write("Definition of terms:\n")
        f.write("Stream Thrust = Mdot*Vel + Pexit*Aexit\n")
        f.write("Gross Thrust = Mdot*Vel + Pexit*Aexit - Pamb*Aexit\n")
        f.write("Stream Thrust = Gross Thrust when Pamb = 0 (vacuum)\n")
        f.write("Cfg = Nozzle Thrust Coefficient (Gross Isp/Ideal Isp)\n")
        f.write("C* = Characteristic Exhaust Velocity\n")
        f.write("CD = Discharge coefficient: Massflow 2D/ Massflow 1D\n")
        f.write("1D = Assumes input are axial\n")
        f.write("2D = Uses actual flow angles in calculation\n\n")
        f.write("Input parameters\nR*  = Throat Radius (in)\n\nNozzle Type: ")
        if n_type == RAO:
            f.write("Minimum Length Nozzle")
        elif n_type == CONE:
            f.write(f"Cone, with half angle (deg) = {_fmt(design_value[1])}")
        elif n_type == PERFECT:
            f.write("Perfect Nozzle")
        elif n_type == FIXEDEND:
            f.write("Nozzle with a fixed end point, X/R* = "
                    f"{_fmt(design_value[0])}\tR/R* = {_fmt(design_value[1])}")
        f.write("\n\nNozzle Geometry:\t")
        if geom == TWOD:
            f.write("Two dimensional Nozzle (assumes 1ft reference width)")
        elif geom == AXI:
            f.write("Axisymmetric Nozzle")
        f.write("\nDesign Parameter:\t")
        if design_param == EXITMACH:
            f.write(f"Exit Mach Number = {_fmt(design_value[0])}")
        elif design_param == EPS:
            f.write(f"Nozzle Area Ratio = {_fmt(design_value[0])}")
        elif design_param == NOZZLELENGTH:
            f.write(f"Nozzle Length = {_fmt(design_value[0])}")
        elif design_param == EXITPRESSURE:
            f.write(f"PTotal/PExit = {_fmt(design_value[0])}")

        f.write("\nThroat Geometry created by two arcs of given radii."
                f"\nUpstream Radius/R*:\t{_fmt(result.rwt_u)}"
                f"\nDownstream Radius/R*:\t{_fmt(result.rwt_d)}"
                "\n\nThermodynamic Inputs"
                f"\nTotal Pressure (psia):\t{_fmt(p_total)}"
                f"\nTotal Temperature (R):\t{_fmt(t_total)}"
                f"\nGamma:\t{_fmt(gamma_i)}"
                f"\nMolecular Weight:\t{_fmt(mol_wt)}"
                f"\nAmbient Pressure (psia):\t{_fmt(p_ambient)}"
                "\n\nMOC Grid Parameters"
                f"\nNumber of characteristic lines at the nozzle throat:\t{result.n_c}"
                "\nMaximum allowable angle between characteristics along wall (deg):\t"
                f"{_fmt(result.dt_limit_rad * DEG_PER_RAD)}"
                "\nNumber of RRCs to calculate above KERNEL (50-100):\t"
                f"{result.n_rrc_above_bd}")

        # --- initial data line --------------------------------------
        f.write("\n\nThe Initial Data Line is where the MOC calculation begins.\n"
                "Its based on the upstream radius of curvature and gamma.\n"
                "It is solved for using a method developed\n"
                "by Kliegel and Levine. This method is based on a method\n"
                "developed by Hall. Hall used cylindrical coordinates (R)\n"
                "This method uses toroid coordinates (R+1). This method\n"
                "does better for small upstream radii.\n\n"
                "Initial Data Line (Throat)\n\n"
                "I\tX/R*\tR/R*\tMach\tFlow Angle(deg)\tPres.(psia)\tTemp.(R)\t"
                "Den.(slug/ft3)\tGamma\tMass Flow(lbm/s)\n")
        for i in range(int(g.i_last[0]) + 1):
            f.write(_row(i, g.x[i, 0], g.r[i, 0], g.mach[i, 0],
                          g.theta[i, 0] * DEG_PER_RAD,
                          g.pres[i, 0], g.temp[i, 0],
                          g.rho[i, 0], g.gamma[i, 0],
                          g.massflow[i, 0]) + "\n")

        # --- initial throat plane summary ---------------------------
        f.write("\nInitial Data Line(Throat Plane) \nFor a throat radius(in) = 1.0. ")
        if geom == TWOD:
            f.write("Values are for half the nozzle with 12in reference width.")
        f.write(f"\n2-D Mass Flow (lbm/s):\t{_fmt(g.massflow[0, 0])}"
                f"\n2-D Gross Thrust (lbf):\t{_fmt(g.thrust[0, 0])}"
                f"\n2-D Isp (lbf-s/lbm):\t"
                f"{_fmt(g.thrust[0, 0] / g.massflow[0, 0]) if g.massflow[0, 0] else '0'}")

        # 1-D throat values
        pt_iso = isentropic_p_t_rho(p_total, t_total, mol_wt, gamma_i, 1.0)
        p_star = pt_iso.pressure
        t_star = pt_iso.temperature
        rho_star = pt_iso.density
        f_1d = (p_star - p_ambient) * (1.0 + gamma_i)
        mdot_1d = GRAV * p_star * math.sqrt(
            gamma_i / (GASCON * GRAV / mol_wt * t_star)
        )
        if geom == TWOD:
            f.write("\nFor a 2-D Nozzle, thrust and massflow are proportional to the throat radius."
                    "\nTo find these values for a radius other than 1 inch, multiple the values by"
                    "\nthe new radius. Make sure it also is in inches.")
            c_star = (12.0 * p_total * GRAV / g.massflow[0, 0]
                      if g.massflow[0, 0] else 0.0)
            f_1d *= 12.0
            mdot_1d *= 12.0
            c_star_1d = 12.0 * p_total * GRAV / mdot_1d
        else:
            f.write("\nFor an axisymmeric nozzle ,thrust and massflow are proportional to the throat"
                    "\nradius squared (R*^2). To find these values for a radius other than 1 inch,"
                    "\nmultiple the values by the square of the new radius. For example, for a radius"
                    "\nof 3 inches, thrust and massflow would increase by a factor of 9.")
            c_star = (p_total * PI * GRAV / g.massflow[0, 0]
                      if g.massflow[0, 0] else 0.0)
            f_1d *= PI
            mdot_1d *= PI
            c_star_1d = p_total * PI * GRAV / mdot_1d
        f.write(f"\n2-D C* (ft/s):\t{_fmt(c_star)}\n")
        f.write("\nBased on the total conditions and assuming Mach = 1 at throat"
                f"\n1-D Mass Flow (lbm/s):\t{_fmt(mdot_1d)}"
                f"\n1-D Gross Thrust (lbf):\t{_fmt(f_1d)}"
                f"\n1-D Gross Isp (lbf-s/lbm):\t{_fmt(f_1d / mdot_1d)}"
                f"\n1-D C* (ft/s):\t{_fmt(c_star_1d)}"
                f"\nCD(2-D/1-D):\t{_fmt(g.massflow[0, 0] / mdot_1d)}")

        # --- ThetaB / kernel error ----------------------------------
        f.write(f"\n\nInitial Expansion Angle ThetaB(deg):\t{_fmt(result.theta_b_ans * DEG_PER_RAD)}"
                f"\nMassflow error due to grid at end of Kernel (%):\t"
                f"{_fmt((1.0 - result.mdot_err_ratio) * 100)}")

        # --- LRC DE line --------------------------------------------
        if n_type != CONE:
            f.write("\n\nLast LRC, DE that effects the nozzle wall contour."
                    "\nJ\tX/R*\tR/R*\tMach\tTheta(deg)\tP(psia)\tT(R)\t"
                    "Velocity(ft/s)\tDensity(slug/ft3)\tGamma\t\n")
            for j in range(result.j_de_last + 1):
                f.write(_row(j, g.x_de[j], g.r_de[j], g.m_de[j],
                              g.theta_de[j] * DEG_PER_RAD,
                              g.p_de[j], g.t_de[j], g.rho_de[j],
                              g.g_de[j]) + "\n")

        # --- wall contour -------------------------------------------
        f.write("\n\nCalculated wall contour"
                "\nJ\tX/R*\tR/R*\tMach\tTheta(deg)\tP(psia)\tT(R)\t"
                "Density(slug/ft3)\tGamma\tMass Flow (lbm)\t%Dif in Mdot\n")
        for j in range(last_rrc + 1):
            pct = ((g.massflow[0, j] - g.massflow[0, 0]) / g.massflow[0, 0] * 100
                   if g.massflow[0, 0] else 0.0)
            f.write(_row(j, g.x[0, j], g.r[0, j], g.mach[0, j],
                          g.theta[0, j] * DEG_PER_RAD,
                          g.pres[0, j], g.temp[0, j], g.rho[0, j],
                          g.gamma[0, j], g.massflow[0, j], pct) + "\n")

        # --- nozzle summary -----------------------------------------
        f.write("\n\nNozzle Summary"
                "\nThroat Radius R* (in):\t1.000"
                f"\nNozzle Length/R*:\t{_fmt(g.x[0, last_rrc])}"
                "\nExpansion Ratio:\t")
        if geom == TWOD:
            f.write(_fmt(g.r[0, last_rrc]))
        else:
            f.write(_fmt(g.r[0, last_rrc] ** 2))
        f.write("\nExit Area (in2):\t")
        if geom == TWOD:
            f.write(_fmt(12.0 * g.r[0, last_rrc]))
        else:
            f.write(_fmt(PI * g.r[0, last_rrc] ** 2))
        f.write(f"\nSurface Area (in2):\t{_fmt(_calc_surface_area(g, last_rrc, geom))}")

        # --- exit plane profile + performance -----------------------
        f.write("\n\nNozzle Exit Plane Data"
                "\nJ\tX/R*\tR/R*\tMach\tTheta(deg)\tP(psia)\tT(R)\t"
                "Velocity(ft/s)\tDensity(slug/ft3)\tGamma\t"
                "Mass Flow (lbm/s)\tStream Thrust(lbf)\n")
        jj = 0
        while (jj <= last_rrc
               and g.x[g.i_last[jj], jj] < g.x[0, last_rrc]):
            jj += 1
        if jj > last_rrc:
            raise RuntimeError("There was an error finding the nozzle exit plane")

        mdot_total = 0.0
        f_total = 0.0
        for j in range(jj, last_rrc + 1):
            iLj = int(g.i_last[j])
            vel2 = g.mach[iLj, j] * math.sqrt(
                g.gamma[iLj, j] * GASCON / mol_wt * GRAV * g.temp[iLj, j])
            row_prefix = _row(j, g.x[iLj, j], g.r[iLj, j], g.mach[iLj, j],
                               g.theta[iLj, j] * DEG_PER_RAD,
                               g.pres[iLj, j], g.temp[iLj, j],
                               vel2, g.rho[iLj, j], g.gamma[iLj, j])
            if j == jj:
                mdot_total = 0.0
                f_total = 0.0
            else:
                iLm = int(g.i_last[j-1])
                if geom == TWOD:
                    da = 12.0 * (g.r[iLj, j] - g.r[iLm, j-1])
                else:
                    da = (PI * (g.r[iLj, j]**2 - g.r[iLm, j-1]**2) / 144.0)
                vel1 = g.mach[iLm, j-1] * math.sqrt(
                    g.gamma[iLm, j-1] * GASCON / mol_wt * GRAV * g.temp[iLm, j-1])
                rho_u_avg = 0.5 * (
                    g.rho[iLj, j] * vel2 * math.cos(g.theta[iLj, j])
                    + g.rho[iLm, j-1] * vel1 * math.cos(g.theta[iLm, j-1]))
                vel_avg = (vel2 * math.cos(g.theta[iLj, j])
                           + vel1 * math.cos(g.theta[iLm, j-1])) / 2.0
                p_avg = (g.pres[iLj, j] + g.pres[iLm, j-1]) / 2.0
                mdot_total += rho_u_avg * da * GRAV
                f_total += (rho_u_avg * vel_avg + p_avg * 144.0) * da
            f.write(row_prefix + "\t" + _fmt(mdot_total) + "\t" + _fmt(f_total) + "\n")

        jj_iL = int(g.i_last[jj])
        f.write(f"\nExit Mach at wall:\t{_fmt(g.mach[0, last_rrc])}"
                f"\nExit Mach at centerline:\t{_fmt(g.mach[jj_iL, jj])}"
                f"\nMassflow (lbm/s):\t{_fmt(mdot_total)}"
                f"\nStream Thrust (lbf):\t{_fmt(f_total)}"
                f"\nIsp (lbf-s/lbm):\t{_fmt(f_total / mdot_total if mdot_total else 0.0)}"
                f"\nIdeal Isp (lbf-s/lbm:\t{_fmt(isp_ideal)}"
                f"\nCfg:\t{_fmt(f_total / mdot_total / isp_ideal if mdot_total and isp_ideal else 0.0)}"
                "\n% diffence in initial and exit mass flow:\t"
                f"{_fmt((mdot_total - g.massflow[0, 0]) / g.massflow[0, 0] * 100 if g.massflow[0, 0] else 0.0)}"
                "\n\nAll of the MOC data can be found in the file MOC_Grid.out"
                "\nThe Streamlines can be found in SL.out"
                "\nThe centerline data can be found in center.out")


def _calc_surface_area(grid, j_last: int, geom: int) -> float:
    sa = 0.0
    for j in range(1, j_last + 1):
        r_avg = 0.5 * (grid.r[0, j] + grid.r[0, j-1])
        length = math.sqrt((grid.r[0, j] - grid.r[0, j-1])**2
                           + (grid.x[0, j] - grid.x[0, j-1])**2)
        if geom == TWOD:
            sa += length * 12.0
        else:
            sa += length * 2.0 * PI * r_avg
    return sa


# ---------------------------------------------------------------------------
def write_moc_grid(result: MOCResult, path: str = "MOC_Grid.plt") -> None:
    """Direct port of ``MOC_GridCalc::OutputMOC_Grid``.

    Writes a Tecplot-format file with one ``zone`` per RRC. Even on Linux
    this is just a text file -- it can be opened in ParaView,
    matplotlib, or VisIt with a Tecplot reader.
    """
    g = result.grid
    assert g is not None
    with open(path, "w") as f:
        f.write('VARIABLES = "X/R","R/R","Mach","Theta","Massflow","I",\n')
        f.write('TITLE = "RRC Contours"\n')
        f.write('text  x=5  y=93  t="MOC Grid Data"\n')
        for j in range(result.last_rrc + 1):
            f.write(f'zone t="J = {j}"\n')
            for i in range(int(g.i_last[j]) + 1):
                f.write(_row(g.x[i, j], g.r[i, j], g.mach[i, j],
                              g.theta[i, j] * DEG_PER_RAD,
                              g.massflow[i, j], i) + "\n")


# ---------------------------------------------------------------------------
def write_center(result: MOCResult, path: str = "center.out") -> None:
    """Direct port of ``MOC_GridCalc::OutputCenterlineData``."""
    g = result.grid
    assert g is not None
    with open(path, "w") as f:
        f.write("Centerline Data file\n")
        f.write("J\tX/R*\tR/R*\tMach\tPres\tTemp\tRho\tTheta\tGamma\tMassFlow\n")
        for j in range(result.last_rrc + 1):
            i = int(g.i_last[j])
            if g.r[i, j] == 0.0:
                f.write(_row(j, g.x[i, j], g.r[i, j], g.mach[i, j],
                              g.pres[i, j], g.temp[i, j], g.rho[i, j],
                              g.theta[i, j] * DEG_PER_RAD,
                              g.gamma[i, j], g.massflow[i, j]) + "\n")


# ---------------------------------------------------------------------------
def write_rao_dat(result: MOCResult, path: str = "rao.dat") -> None:
    """Direct port of ``MOC_GridCalc::OutputTDKRAODataFile``.

    Three columns: R/R*, X/R*, theta(deg).
    """
    g = result.grid
    assert g is not None
    j_start = result.j_bd
    j_end = result.last_rrc
    with open(path, "w") as f:
        for j in range(j_start, j_end + 1):
            f.write(_row(g.r[0, j], g.x[0, j],
                          g.theta[0, j] * DEG_PER_RAD) + "\n")


# ---------------------------------------------------------------------------
def write_streamlines(
    result: MOCResult, path: str = "MOC_SL.plt",
    n_sl_i: int | None = None, n_sl_j: int | None = None,
) -> None:
    """Direct port of ``MOC_GridCalc::OutputStreamlines``.

    Writes the streamlines used by the streamline-tracing tool. The
    centerline is the first zone, then ``n_sl_i - 1`` zones at equally-
    spaced mass-flow fractions, then the wall contour.
    """
    g = result.grid
    assert g is not None
    if n_sl_i is None:
        n_sl_i = result.n_sl_i
    if n_sl_j is None:
        n_sl_j = result.n_sl_j
    j_end = result.last_rrc
    geom = result.nozzle_geom
    j_step = max(int(j_end / n_sl_j), 1)

    with open(path, "w") as f:
        f.write('VARIABLES = "X/R","Y/R","Z/R","R/R","Mach","Pres","Temp",'
                '"Rho","Theta","Gamma","Massflow""J"\n')
        f.write('TITLE = "Streamline Data file"\n')
        f.write('text  x=5  y=93   t="Streamline Data"\n')

        # --- centerline zone --------------------------------------
        j_cur = -j_step
        count = 1
        while j_cur < j_end - j_step:
            j_cur += j_step
            i = int(g.i_last[j_cur])
            if g.r[i, j_cur] != 0.0:
                jj = j_cur
                while g.r[int(g.i_last[jj]), jj] != 0.0:
                    jj -= 1
                j_cur = jj
                break
            count += 1
        f.write(f'zone t="MassFlow % = 0" I = {count} J = 1 K = 1\n')
        j_cur = -j_step
        while j_cur < j_end - j_step:
            j_cur += j_step
            i = int(g.i_last[j_cur])
            if g.r[i, j_cur] == 0.0:
                f.write(_row(g.x[i, j_cur], 0.0, 0.0, g.r[i, j_cur],
                              g.mach[i, j_cur], g.pres[i, j_cur],
                              g.temp[i, j_cur], g.rho[i, j_cur],
                              g.theta[i, j_cur] * DEG_PER_RAD,
                              g.gamma[i, j_cur], g.massflow[i, j_cur],
                              j_cur) + "\n")
            else:
                while g.r[int(g.i_last[j_cur]), j_cur] != 0.0:
                    j_cur -= 1
                i = int(g.i_last[j_cur])
                f.write(_row(g.x[i, j_cur], 0.0, 0.0, g.r[i, j_cur],
                              g.mach[i, j_cur], g.pres[i, j_cur],
                              g.temp[i, j_cur], g.rho[i, j_cur],
                              g.theta[i, j_cur] * DEG_PER_RAD,
                              g.gamma[i, j_cur], g.massflow[i, j_cur],
                              j_cur) + "\n")
                break

        # --- interior streamlines, equally spaced in mass flow -----
        mdot_match = 0.0
        if g.massflow[0, 0] <= 0.0:
            # nothing to do; just write the wall and bail
            _write_wall_zone(f, g, j_end, j_step, geom)
            return
        mdot_step = g.massflow[0, 0] / n_sl_i

        for i_zone in range(1, n_sl_i):
            mdot_match += mdot_step
            # determine point count
            j_cur = -j_step
            count = 0
            j_for_block = j_cur
            while j_for_block < j_end:
                j_for_block += j_step
                if j_for_block > j_end:
                    j_for_block = j_end
                im = int(g.i_last[j_for_block])
                while g.massflow[im, j_for_block] <= mdot_match and im > 0:
                    im -= 1
                if (im == g.i_last[j_for_block]
                        and g.massflow[im, j_for_block] > mdot_match):
                    while g.massflow[int(g.i_last[j_for_block]), j_for_block] > mdot_match:
                        j_for_block -= 1
                    j_for_block += 1
                    j_for_block = j_end + 1
                count += 1

            f.write(f'zone t="MassFlow % = {_fmt(mdot_match / g.massflow[0, 0] * 100)}"'
                    f' I = {count} J = 1 K = 37 \n')

            for k in range(37):
                j_cur = -j_step
                while j_cur < j_end:
                    j_cur += j_step
                    if j_cur > j_end:
                        j_cur = j_end
                    im = int(g.i_last[j_cur])
                    while g.massflow[im, j_cur] <= mdot_match and im > 0:
                        im -= 1
                    if (im == g.i_last[j_cur]
                            and g.massflow[im, j_cur] > mdot_match):
                        while g.massflow[int(g.i_last[j_cur]), j_cur] > mdot_match:
                            j_cur -= 1
                        j_cur += 1
                        ratio = ((mdot_match - g.massflow[int(g.i_last[j_cur-1]), j_cur-1])
                                 / (g.massflow[int(g.i_last[j_cur]), j_cur]
                                    - g.massflow[int(g.i_last[j_cur-1]), j_cur-1]))
                        iLj  = int(g.i_last[j_cur])
                        iLjm = int(g.i_last[j_cur-1])
                        def interp(arr):
                            return arr[iLjm, j_cur-1] + ratio * (arr[iLj, j_cur] - arr[iLjm, j_cur-1])
                        mdot = interp(g.massflow)
                        M  = interp(g.mach)
                        T  = interp(g.temp)
                        P  = interp(g.pres)
                        A  = interp(g.theta)
                        D  = interp(g.rho)
                        R  = interp(g.r)
                        X  = interp(g.x)
                        G  = interp(g.gamma)
                        F  = interp(g.thrust)
                        j_cur = j_end + 1
                    elif (im == g.i_last[j_cur]
                            and g.massflow[im, j_cur] < mdot_match):
                        M = g.mach[im, j_cur]
                        T = g.temp[im, j_cur]
                        P = g.pres[im, j_cur]
                        A = g.theta[im, j_cur]
                        D = g.rho[im, j_cur]
                        R = g.r[im, j_cur]
                        X = g.x[im, j_cur]
                        G = g.gamma[im, j_cur]
                        F = g.thrust[im, j_cur]
                        mdot = g.massflow[im, j_cur]
                    else:
                        if g.massflow[im, j_cur] < mdot_match and im == 0:
                            ratio = 0.0
                        else:
                            ratio = ((mdot_match - g.massflow[im+1, j_cur])
                                     / (g.massflow[im, j_cur] - g.massflow[im+1, j_cur]))
                        def interp(arr):
                            return arr[im+1, j_cur] + ratio * (arr[im, j_cur] - arr[im+1, j_cur])
                        mdot = interp(g.massflow)
                        M    = interp(g.mach)
                        T    = interp(g.temp)
                        P    = interp(g.pres)
                        A    = interp(g.theta)
                        D    = interp(g.rho)
                        R    = interp(g.r)
                        X    = interp(g.x)
                        G    = interp(g.gamma)
                        F    = interp(g.thrust)
                    if geom == AXI:
                        Y = R * math.sin(k * PI / 18.0)
                        Z = R * math.cos(k * PI / 18.0)
                    else:
                        Y = R
                        Z = k / 3.0
                    f.write(_row(X, Y, Z, R, M, P, T, D, A * DEG_PER_RAD,
                                  G, mdot, j_cur) + "\n")
        _write_wall_zone(f, g, j_end, j_step, geom)


def _write_wall_zone(f, grid, j_end: int, j_step: int, geom: int) -> None:
    """Write the wall contour as the final zone of MOC_SL.plt."""
    i = 0
    count = 0
    j_cur = -j_step
    while j_cur < j_end:
        j_cur += j_step
        if j_cur > j_end:
            j_cur = j_end
        count += 1
    f.write(f'zone t="MassFlow % = 100" I = {count} J = 1 K = 37\n')
    for k in range(37):
        j_cur = -j_step
        while j_cur < j_end:
            j_cur += j_step
            if j_cur > j_end:
                j_cur = j_end
            R = grid.r[i, j_cur]
            if geom == AXI:
                Y = R * math.sin(k * PI / 18.0)
                Z = R * math.cos(k * PI / 18.0)
            else:
                Y = R
                Z = k / 3.0
            f.write(_row(grid.x[i, j_cur], Y, Z, grid.r[i, j_cur],
                          grid.mach[i, j_cur], grid.pres[i, j_cur],
                          grid.temp[i, j_cur], grid.rho[i, j_cur],
                          grid.theta[i, j_cur] * DEG_PER_RAD,
                          grid.gamma[i, j_cur], grid.massflow[i, j_cur],
                          j_cur) + "\n")


# ---------------------------------------------------------------------------
def write_all(
    result: MOCResult, output_dir: str | Path = ".",
    full_output: bool = True,
) -> None:
    """Write all output files appropriate to the print mode.

    With ``full_output=True`` this writes ``summary.out``, ``MOC_Grid.plt``,
    ``MOC_SL.plt``, ``center.out``, and ``rao.dat``. With ``False`` only
    the summary and rao.dat are written, mimicking the C++ behavior when
    ``printMode == 0``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(result, str(output_dir / "summary.out"))
    if result.nozzle_type != CONE:
        write_rao_dat(result, str(output_dir / "rao.dat"))
    if full_output:
        write_moc_grid(result, str(output_dir / "MOC_Grid.plt"))
        write_center(result, str(output_dir / "center.out"))
        write_streamlines(result, str(output_dir / "MOC_SL.plt"))
