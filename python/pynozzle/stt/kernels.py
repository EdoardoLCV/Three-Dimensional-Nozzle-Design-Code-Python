"""Computation kernels for STT2001 -- direct ports of the corresponding
methods in ``CSTT2001Dlg``.

Each function takes a :class:`STTState` (defined in :mod:`solver`) and
mutates it the same way the C++ class mutates its members. The methods
are kept as free functions rather than methods on the state object to
keep the file boundary at a natural seam between numerical kernels and
the orchestration logic in :mod:`solver`.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ..common.constants import PI, DEG_PER_RAD, RAD_PER_DEG

if TYPE_CHECKING:
    from .solver import STTState


def _fmt(x: float) -> str:
    """Match the default C++ ostream output (up to 6 sig figs)."""
    if x != x:
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
# CalcThroatSLs                                                              -
# ---------------------------------------------------------------------------
def calc_throat_sls(
    state: "STTState",
    yc: float, zc: float, rc: float,
    alpha: float, omega: float, n_theta_sls: int,
    i_surface: int,
) -> bool:
    """Direct port of ``CSTT2001Dlg::CalcThroatSLs``.

    Walks angles ``theta = alpha + p * dTheta`` around the throat
    boundary; for each one finds the SL in the loaded ``MOC_SL.plt``
    whose ``rsl[i][0]`` equals the requested constraint radius, then
    constructs a new "throat SL" (the ``xt``, ``yt``, ``zt``, ``pt``
    arrays) by interpolating between adjacent loaded SLs.

    Returns ``True`` on success, ``False`` if any throat SL could not
    be found.
    """
    s = state
    s.circle_flag = 0
    if abs(omega - alpha) >= 360:
        alpha = 0.0
        d_theta = 360.0 / n_theta_sls
        omega = 360.0 - d_theta
        s.circle_flag = 1
    else:
        d_theta = (omega - alpha) / (n_theta_sls - 1)

    n = s.n_new_sls - 1

    for p in range(n_theta_sls):
        theta = alpha + p * d_theta
        n += 1

        if i_surface == 0:
            i_surface_eff = -1
        else:
            i_surface_eff = i_surface
        if d_theta != abs(d_theta):
            s.rotate_sl[n] = i_surface_eff
        else:
            s.rotate_sl[n] = -i_surface_eff

        # Walk i over loaded SLs to find the bracket.
        found = False
        for i in range(s.sl.n_sl):
            y = rc * math.cos(theta * RAD_PER_DEG) + yc
            z = rc * math.sin(theta * RAD_PER_DEG) + zc
            beta = math.atan2(z - s.m_ZSL, y - s.m_YSL)
            r = math.sqrt((y - s.m_YSL) ** 2 + (z - s.m_ZSL) ** 2)

            if r <= s.sl.rsl[i, 0] * s.m_RSL:
                if i > 0:
                    dr = ((s.sl.rsl[i, 0] - r / s.m_RSL)
                          / (s.sl.rsl[i, 0] - s.sl.rsl[i - 1, 0]))
                    s.new_psi[n] = s.sl.psi[i] - dr * (s.sl.psi[i] - s.sl.psi[i - 1])
                else:
                    dr = (s.sl.rsl[i, 0] - r / s.m_RSL) / s.sl.rsl[i, 0]
                    s.new_psi[n] = s.sl.psi[i] * (1 - dr)

                # First point on the throat plane
                s.xt[n, 0] = s.sl.xsl[i, 0] * s.m_RSL + s.m_XSL
                s.yt[n, 0] = y
                s.zt[n, 0] = z
                s.pt[n, 0] = s.sl.psl[i, 0]
                s.nt[n] = i

                # Walk along the i-th SL, interpolating in x against the
                # (i-1)-th SL to get the y, z, p of the new "throat SL".
                last_j_i = int(s.sl.last_sl_pt[i])
                last_j_im = int(s.sl.last_sl_pt[i - 1]) if i > 0 else 0
                for j in range(1, last_j_i + 1):
                    k = 1
                    if i > 0:
                        # find k such that xsl[i-1][k] >= xsl[i][j]
                        while True:
                            k += 1
                            if (s.sl.xsl[i - 1, k] >= s.sl.xsl[i, j]
                                    or k > last_j_im):
                                break
                        if (k != 1
                                and s.sl.xsl[i - 1, k] != s.sl.xsl[i - 1, k - 1]):
                            dx = ((s.sl.xsl[i - 1, k] - s.sl.xsl[i, j])
                                  / (s.sl.xsl[i - 1, k] - s.sl.xsl[i - 1, k - 1]))
                        else:
                            dx = 0.0
                        r_tmp = (s.sl.rsl[i - 1, k]
                                 - dx * (s.sl.rsl[i - 1, k] - s.sl.rsl[i - 1, k - 1]))
                        p_tmp = (s.sl.psl[i - 1, k]
                                 - dx * (s.sl.psl[i - 1, k] - s.sl.psl[i - 1, k - 1]))
                    else:
                        r_tmp = 0.0
                        p_tmp = s.sl.psl[i, j]

                    rt = (s.sl.rsl[i, j] - dr * (s.sl.rsl[i, j] - r_tmp)) * s.m_RSL
                    s.pt[n, j] = (s.sl.psl[i, j]
                                  - dr * (s.sl.psl[i, j] - p_tmp))
                    s.xt[n, j] = s.sl.xsl[i, j] * s.m_RSL + s.m_XSL
                    s.yt[n, j] = rt * math.cos(beta) + s.m_YSL
                    s.zt[n, j] = rt * math.sin(beta) + s.m_ZSL
                    s.new_sl_pts[n] = j
                    if j == last_j_i:
                        s.xt_end[n] = s.xt[n, j]
                        s.yt_end[n] = s.yt[n, j]
                        s.zt_end[n] = s.zt[n, j]
                found = True
                break

        if not found:
            return False

    s.n_new_sls = n + 1
    return True


# ---------------------------------------------------------------------------
# TrimSLs                                                                    -
# ---------------------------------------------------------------------------
def trim_sls(
    state: "STTState",
    yc: float, zc: float, rc: float,
    alpha: float, omega: float,
    x_start: float, x_end: float,
) -> None:
    """Direct port of ``CSTT2001Dlg::TrimSLs`` — cylindrical-constraint trim."""
    s = state
    for n in range(s.n_new_sls):
        last_j = int(s.new_sl_pts[n])
        for j in range(last_j + 1):
            r = math.sqrt((s.yt[n, j] - yc) ** 2 + (s.zt[n, j] - zc) ** 2)
            if r >= rc:
                theta = math.atan2(s.zt[n, j], s.yt[n, j]) * DEG_PER_RAD
                if alpha >= 0.0 and omega >= 0.0 and theta < 0.0:
                    theta = 360.0 + theta
                elif alpha <= 0.0 and omega <= 0.0 and theta > 0.0:
                    theta = theta - 360.0
                lo = min(alpha, omega)
                hi = max(alpha, omega)
                if (s.xt[n, j] >= x_start and s.xt[n, j] <= x_end
                        and lo <= theta <= hi):
                    r_minus = math.sqrt(
                        (s.yt[n, j - 1] - yc) ** 2 + (s.zt[n, j - 1] - zc) ** 2)
                    dr = (r - rc) / (r - r_minus) if (r - r_minus) else 0.0
                    x = s.xt[n, j] - dr * (s.xt[n, j] - s.xt[n, j - 1])
                    if (s.xt[n, j] - s.xt[n, j - 1]) == 0.0:
                        dx = 0.0
                    else:
                        dx = ((s.xt[n, j] - x)
                              / (s.xt[n, j] - s.xt[n, j - 1]))
                    y = s.yt[n, j] - dx * (s.yt[n, j] - s.yt[n, j - 1])
                    z = s.zt[n, j] - dx * (s.zt[n, j] - s.zt[n, j - 1])
                    p = s.pt[n, j] - dx * (s.pt[n, j] - s.pt[n, j - 1])
                    s.xt[n, j] = x
                    s.yt[n, j] = y
                    s.zt[n, j] = z
                    s.pt[n, j] = p
                    s.new_sl_pts[n] = j
                break


# ---------------------------------------------------------------------------
# TrimSLsToMaxLength                                                         -
# ---------------------------------------------------------------------------
def trim_sls_to_max_length(state: "STTState", max_x: float) -> None:
    """Direct port of ``CSTT2001Dlg::TrimSLsToMaxLength``."""
    s = state
    s.n_at_max_x = 0
    for n in range(s.n_new_sls):
        last_j = int(s.new_sl_pts[n])
        for j in range(last_j + 1):
            if s.xt[n, j] > max_x:
                s.n_at_max_x = n
                if j > 0:
                    denom = s.xt[n, j] - s.xt[n, j - 1]
                    dx = ((s.xt[n, j] - max_x) / denom) if denom != 0.0 else 0.0
                    s.xt[n, j] = s.xt[n, j] - dx * (s.xt[n, j] - s.xt[n, j - 1])
                    s.yt[n, j] = s.yt[n, j] - dx * (s.yt[n, j] - s.yt[n, j - 1])
                    s.zt[n, j] = s.zt[n, j] - dx * (s.zt[n, j] - s.zt[n, j - 1])
                    s.pt[n, j] = s.pt[n, j] - dx * (s.pt[n, j] - s.pt[n, j - 1])
                else:
                    s.xt[n, j] = s.yt[n, j] = s.zt[n, j] = s.pt[n, j] = 0.0
                for k in range(j + 1, last_j + 1):
                    s.xt[n, k] = s.xt[n, j]
                    s.yt[n, k] = s.yt[n, j]
                    s.zt[n, k] = s.zt[n, j]
                    s.pt[n, k] = s.pt[n, j]
                break


# ---------------------------------------------------------------------------
# CalcGridSLs                                                                -
# ---------------------------------------------------------------------------
def _divided_difference_5(
    x_arr: list[float], y_arr: list[float], x_eval: float,
) -> float:
    """Newton's divided-difference interpolation across 5 points.

    Mirrors the inline ``yf[i][jj]`` / ``zf[i][jj]`` / ``pf[i][jj]``
    tables built in ``CalcGridSLs`` for use in the 5-point branch.
    """
    N = 4
    f = [[0.0] * 5 for _ in range(5)]
    for jj in range(5):
        f[0][jj] = y_arr[jj]
    for i in range(1, 5):
        for jj in range(N - i + 1):
            f[i][jj] = ((f[i - 1][jj + 1] - f[i - 1][jj])
                        / (x_arr[jj + i] - x_arr[jj]))
    return (f[0][0]
            + f[1][0] * (x_eval - x_arr[0])
            + f[2][0] * (x_eval - x_arr[0]) * (x_eval - x_arr[1])
            + f[3][0] * (x_eval - x_arr[0]) * (x_eval - x_arr[1])
                                            * (x_eval - x_arr[2])
            + f[4][0] * (x_eval - x_arr[0]) * (x_eval - x_arr[1])
                                            * (x_eval - x_arr[2])
                                            * (x_eval - x_arr[3]))


def calc_grid_sls(state: "STTState") -> None:
    """Direct port of ``CSTT2001Dlg::CalcGridSLs``.

    Resamples each trimmed throat SL onto a common axial grid with
    ``nparamGRIDX`` points. Half of the grid points sit in the first
    third of the nozzle to give better resolution near the throat
    where the wall curvature changes fastest.
    """
    s = state
    n_grid_x = s.n_param_grid_x
    x3 = s.x1 + (s.x2 - s.x1) / 3.0

    s.max_length = 0.0
    s.min_y = 9e9
    s.max_y = -9e9

    for n in range(s.n_new_sls):
        last_j = int(s.new_sl_pts[n])
        for k in range(n_grid_x):
            # NOTE: the C++ uses ``(nparamGRIDX-1)/2`` which is *integer*
            # division (== 49 for the default 100-point grid). Python's
            # ``/`` would give 49.5 and produce a ~1.4% systematic error
            # in the downstream area integration, so we force integer
            # division here to match the original.
            half = (n_grid_x - 1) // 2
            if k < n_grid_x // 2:
                x_grid_val = (s.x1 + k * (x3 - s.x1)) * s.m_RSL / half + s.m_XSL
            else:
                x_grid_val = (x3 + k * (s.x2 - x3)) * s.m_RSL / half + s.m_XSL
            s.x_grid[n, k] = x_grid_val

            # Find j such that xt[n, j] >= xgrid[n, k]
            j = 0
            while j <= last_j and s.xt[n, j] < x_grid_val:
                j += 1

            if j == 0:
                s.y_grid[n, k] = s.yt[n, 0]
                s.z_grid[n, k] = s.zt[n, 0]
                s.p_grid[n, k] = s.pt[n, 0]
            elif j > last_j:
                if k > 0:
                    s.x_grid[n, k] = s.xt[n, j - 1]
                    s.y_grid[n, k] = s.yt[n, j - 1]
                    s.z_grid[n, k] = s.zt[n, j - 1]
                    s.p_grid[n, k] = s.pt[n, j - 1]
                else:
                    s.x_grid[n, k] = s.xt[n, j]
                    s.y_grid[n, k] = s.yt[n, j]
                    s.z_grid[n, k] = s.zt[n, j]
                    s.p_grid[n, k] = s.pt[n, j]
            elif j + 2 > last_j or j == 1:
                # linear
                denom = s.xt[n, j] - s.xt[n, j - 1]
                dx = ((s.xt[n, j] - x_grid_val) / denom) if denom != 0.0 else 0.0
                s.y_grid[n, k] = s.yt[n, j] - dx * (s.yt[n, j] - s.yt[n, j - 1])
                s.z_grid[n, k] = s.zt[n, j] - dx * (s.zt[n, j] - s.zt[n, j - 1])
                s.p_grid[n, k] = s.pt[n, j] - dx * (s.pt[n, j] - s.pt[n, j - 1])
            else:
                # 5-point divided difference
                xf = [s.xt[n, j - 2], s.xt[n, j - 1], s.xt[n, j],
                      s.xt[n, j + 1], s.xt[n, j + 2]]
                yf = [s.yt[n, j - 2], s.yt[n, j - 1], s.yt[n, j],
                      s.yt[n, j + 1], s.yt[n, j + 2]]
                zf = [s.zt[n, j - 2], s.zt[n, j - 1], s.zt[n, j],
                      s.zt[n, j + 1], s.zt[n, j + 2]]
                pf = [s.pt[n, j - 2], s.pt[n, j - 1], s.pt[n, j],
                      s.pt[n, j + 1], s.pt[n, j + 2]]
                try:
                    s.y_grid[n, k] = _divided_difference_5(xf, yf, x_grid_val)
                    s.z_grid[n, k] = _divided_difference_5(xf, zf, x_grid_val)
                    s.p_grid[n, k] = _divided_difference_5(xf, pf, x_grid_val)
                except ZeroDivisionError:
                    s.y_grid[n, k] = float("inf")
                    s.z_grid[n, k] = float("inf")
                    s.p_grid[n, k] = float("inf")
                if not math.isfinite(s.y_grid[n, k]):
                    denom = s.xt[n, j] - s.xt[n, j - 1]
                    dx = ((s.xt[n, j] - x_grid_val) / denom) if denom else 0.0
                    s.y_grid[n, k] = s.yt[n, j] - dx * (s.yt[n, j] - s.yt[n, j - 1])
                    s.z_grid[n, k] = s.zt[n, j] - dx * (s.zt[n, j] - s.zt[n, j - 1])
                    s.p_grid[n, k] = s.pt[n, j] - dx * (s.pt[n, j] - s.pt[n, j - 1])
            if s.p_grid[n, k] < 0.0:
                denom = s.xt[n, j] - s.xt[n, j - 1] if j > 0 else 1.0
                dx = ((s.xt[n, j] - x_grid_val) / denom) if denom else 0.0
                s.y_grid[n, k] = s.yt[n, j] - dx * (s.yt[n, j] - s.yt[n, j - 1])
                s.z_grid[n, k] = s.zt[n, j] - dx * (s.zt[n, j] - s.zt[n, j - 1])
                s.p_grid[n, k] = s.pt[n, j] - dx * (s.pt[n, j] - s.pt[n, j - 1])
                if s.p_grid[n, k] < 0.0:
                    raise RuntimeError(
                        "Calculated grid pressure < 0.0 in calc_grid_sls "
                        f"at n={n}, k={k}")

    # Track exit-plane extrema
    old_max = s.max_y
    old_min = s.min_y
    for n in range(s.n_new_sls):
        x_end = s.x_grid[n, n_grid_x - 1]
        if x_end > s.max_length:
            s.max_length = x_end
            s.n_at_max_x = n
        y_end = s.y_grid[n, n_grid_x - 1]
        if y_end > s.max_y:
            s.max_y = y_end
        if y_end < s.min_y:
            s.min_y = y_end
        if old_max != s.max_y:
            s.x_at_max_y = x_end
            s.n_at_max_y = n
        if old_min != s.min_y:
            s.x_at_min_y = x_end
            s.n_at_min_y = n
        old_max = s.max_y
        old_min = s.min_y


# ---------------------------------------------------------------------------
# TrimSLsDueToAxiRevolution                                                  -
# ---------------------------------------------------------------------------
def trim_sls_due_to_axi_revolution(
    state: "STTState",
    n_revs: int, y_sim: float, z_sim: float, r_sim: float,
) -> None:
    """Direct port of ``CSTT2001Dlg::TrimSLsDueToAxiRevolution``.

    Used when several copies of the nozzle are placed around a common
    line of symmetry; trims away any portion of the streamlines that
    would interfere with the next copy.
    """
    s = state
    n_grid_x = s.n_param_grid_x

    s.max_length = 0.0
    s.min_y = 9e9
    s.max_y = -9e9

    y_trim = s.y_grid.copy()
    z_trim = s.z_grid.copy()

    for i in range(1, n_revs):
        theta_rot = -i * 2 * PI / n_revs
        for n in range(s.n_new_sls):
            j = 1
            while j < n_grid_x:
                y2 = s.y_grid[n, j] - y_sim
                z2 = s.z_grid[n, j] - z_sim
                theta2 = math.atan2(y2, z2)
                rad2 = math.sqrt(y2 * y2 + z2 * z2)

                # Determine matched SL on the symmetric side
                if n < s.sym_sl2_match:
                    isl_match = s.sym_sl2_match - (n + s.sym_sl1_match)
                else:
                    isl_match = s.n_new_sls - n + s.sym_sl2_match - 1

                if s.circle_flag:
                    isl_match = (s.sym_sl1_match - n) if s.sym_sl1_match > 0 else 0
                    if isl_match >= s.sym_sl1_match:
                        isl_match = 0
                isl_match = max(0, min(s.n_new_sls - 1, isl_match))

                cut = False
                for k in (j, j + 1):
                    if k >= n_grid_x:
                        continue
                    y3 = y_trim[isl_match, k] - y_sim
                    z3 = z_trim[isl_match, k] - z_sim
                    theta3 = math.atan2(y3, z3) + theta_rot
                    rad3 = math.sqrt(y3 * y3 + z3 * z3)
                    if theta3 > 2 * PI:
                        theta3 -= 2 * PI
                    elif theta3 < -2 * PI:
                        theta3 += 2 * PI
                    y4 = y_trim[isl_match, k - 1] - y_sim
                    z4 = z_trim[isl_match, k - 1] - z_sim
                    theta4 = math.atan2(y4, z4) + theta_rot
                    if theta4 > 2 * PI:
                        theta4 -= 2 * PI
                    elif theta4 < -2 * PI:
                        theta4 += 2 * PI
                    rad4 = math.sqrt(y4 * y4 + z4 * z4)
                    if abs(theta3 - theta4) > PI:
                        if theta4 >= 0.0:
                            theta4 -= 2 * PI
                        else:
                            theta4 += 2 * PI
                    if theta2 > 0.0 and theta3 < 0.0 and theta4 < 0.0:
                        theta2 -= 2 * PI
                    elif theta2 < 0.0 and theta3 > 0.0 and theta4 > 0.0:
                        theta2 += 2 * PI
                    if theta3 > theta4:
                        theta3, theta4 = theta4, theta3
                        rad3, rad4 = rad4, rad3
                    d_theta = 0.5 * (theta4 - theta3)
                    d_rad = 0.5 * abs(rad4 - rad3)
                    if (theta3 - d_theta <= theta2 <= theta4 + d_theta
                            and min(rad3, rad4) - d_rad
                            <= rad2 <= max(rad3, rad4) + d_rad):
                        for m in range(j, n_grid_x):
                            s.x_grid[n, m] = s.x_grid[n, j]
                            s.y_grid[n, m] = s.y_grid[n, j]
                            s.z_grid[n, m] = s.z_grid[n, j]
                            s.p_grid[n, m] = s.p_grid[n, j]
                        cut = True
                        break
                if cut:
                    break
                j += 1

    old_max = s.max_y
    old_min = s.min_y
    for n in range(s.n_new_sls):
        x_end = s.x_grid[n, n_grid_x - 1]
        if x_end > s.max_length:
            s.max_length = x_end
            s.n_at_max_x = n
        y_end = s.y_grid[n, n_grid_x - 1]
        if y_end > s.max_y:
            s.max_y = y_end
        if y_end < s.min_y:
            s.min_y = y_end
        if old_max != s.max_y:
            s.x_at_max_y = x_end
            s.n_at_max_y = n
        if old_min != s.min_y:
            s.x_at_min_y = x_end
            s.n_at_min_y = n
        old_max = s.max_y
        old_min = s.min_y


# ---------------------------------------------------------------------------
# CalcNozzleParameters                                                       -
# ---------------------------------------------------------------------------
def _tri_vector_area(P1, P2, P3) -> tuple[float, float]:
    """Return (area, W_x) for the triangle (P1,P2,P3) where W = U x V."""
    U = (P2[0] - P1[0], P2[1] - P1[1], P2[2] - P1[2])
    V = (P3[0] - P1[0], P3[1] - P1[1], P3[2] - P1[2])
    Wx = U[1] * V[2] - U[2] * V[1]
    Wy = U[2] * V[0] - U[0] * V[2]
    Wz = U[0] * V[1] - U[1] * V[0]
    mag = math.sqrt(Wx * Wx + Wy * Wy + Wz * Wz)
    return 0.5 * mag, Wx


def calc_nozzle_parameters(state: "STTState", a_vs_x_path: str | None = None,
                           a_vs_sl_path: str | None = None) -> None:
    """Direct port of ``CSTT2001Dlg::CalcNozzleParameters``.

    Triangulates each (n, k) -> (n+1, k+1) cell of the resampled grid
    into two triangles, computes their surface area and axial
    projection, and accumulates pressure-force and surface-area
    integrals. Writes ``AvsX`` and ``AvsSL`` running totals to disk if
    output paths are supplied (the C++ always writes them).
    """
    s = state
    n_grid_x = s.n_param_grid_x

    # Reference pressure ceiling (the C++ uses 1.05 * max throat psl)
    p_max = max(s.sl.psl[i, 0] for i in range(s.n_new_sls + 1)
                if i < s.sl.psl.shape[0])
    p_max *= 1.05

    # X-running totals
    f_avs_x = open(a_vs_x_path, "w") if a_vs_x_path else None
    if f_avs_x:
        f_avs_x.write("X \t Area \t Surface Area \t Pressure Force\n")

    def _proc(P1, P2, P3, P4, p_avg0, p_avg1,
              p1_ok, p2_ok, p3_ok, p4_ok, rot,
              sa_acc, area_acc, force_acc):
        if p1_ok and p2_ok and p3_ok:
            area, Wx = _tri_vector_area(P1, P2, P3)
            sa_acc += area
            mag = 2.0 * area
            if mag != 0.0:
                cos_theta = rot * Wx / mag
                area_acc += area * cos_theta
                force_acc += p_avg0 * area * cos_theta
        if p3_ok and p2_ok and p4_ok:
            area, Wx = _tri_vector_area(P4, P3, P2)
            sa_acc += area
            mag = 2.0 * area
            if mag != 0.0:
                cos_theta = rot * Wx / mag
                area_acc += area * cos_theta
                force_acc += p_avg1 * area * cos_theta
        return sa_acc, area_acc, force_acc

    sa_acc = 0.0
    area_acc = 0.0
    force_acc = 0.0
    for k in range(n_grid_x - 1):
        for n in range(s.n_new_sls):
            P1 = (s.x_grid[n, k],     s.y_grid[n, k],     s.z_grid[n, k])
            P2 = (s.x_grid[n, k + 1], s.y_grid[n, k + 1], s.z_grid[n, k + 1])
            if n != s.n_new_sls - 1:
                p_avg0 = (s.p_grid[n, k] + s.p_grid[n, k + 1] + s.p_grid[n + 1, k]) / 3.0
                p_avg1 = (s.p_grid[n + 1, k + 1] + s.p_grid[n, k + 1] + s.p_grid[n + 1, k]) / 3.0
                P3 = (s.x_grid[n + 1, k],     s.y_grid[n + 1, k],     s.z_grid[n + 1, k])
                P4 = (s.x_grid[n + 1, k + 1], s.y_grid[n + 1, k + 1], s.z_grid[n + 1, k + 1])
            else:
                p_avg0 = (s.p_grid[n, k] + s.p_grid[n, k + 1] + s.p_grid[0, k]) / 3.0
                p_avg1 = (s.p_grid[0, k + 1] + s.p_grid[n, k + 1] + s.p_grid[0, k]) / 3.0
                P3 = (s.x_grid[0, k],     s.y_grid[0, k],     s.z_grid[0, k])
                P4 = (s.x_grid[0, k + 1], s.y_grid[0, k + 1], s.z_grid[0, k + 1])
            # NOTE (faithful C++ reproduction): the pressure non-zero
            # checks below use the *literal* n+1 row, NOT the wrapped
            # index. On the final sector (n == nNewSLs-1) that is row
            # nNewSLs, which the C++ never fills -- it is zero -- so both
            # triangles of the closing sector are skipped. This means the
            # integrated nozzle is open by one sector, scaling all areas
            # by (nNewSLs-1)/nNewSLs. We reproduce it so our output
            # matches the sample. ``p_grid`` has an extra zero row for
            # exactly this purpose.
            p_n1k_p  = s.p_grid[n + 1, k]
            p_n1k1_p = s.p_grid[n + 1, k + 1]
            rot = s.rotate_sl[n]
            sa_acc, area_acc, force_acc = _proc(
                P1, P2, P3, P4, p_avg0, p_avg1,
                s.p_grid[n, k]     != 0.0,
                s.p_grid[n, k + 1] != 0.0,
                p_n1k_p            != 0.0,
                p_n1k1_p           != 0.0,
                rot, sa_acc, area_acc, force_acc,
            )
        if f_avs_x:
            f_avs_x.write(
                f"{_fmt(s.x_grid[s.n_at_max_x, k + 1])}\t"
                f"{_fmt(area_acc + s.a_throat)}\t{_fmt(sa_acc)}\t{_fmt(force_acc)}\n"
            )
    if f_avs_x:
        f_avs_x.close()
    s.m_surface_area = sa_acc
    s.m_projected_area = area_acc
    s.m_force = force_acc

    # SL-running totals (used as a consistency check)
    f_avs_sl = open(a_vs_sl_path, "w") if a_vs_sl_path else None
    if f_avs_sl:
        f_avs_sl.write("SL# \t Area \t Surface Area \t Pressure Force\n")
    tmp_sa = tmp_area = tmp_force = 0.0
    for n in range(s.n_new_sls):
        for k in range(n_grid_x - 1):
            P1 = (s.x_grid[n, k],     s.y_grid[n, k],     s.z_grid[n, k])
            P2 = (s.x_grid[n, k + 1], s.y_grid[n, k + 1], s.z_grid[n, k + 1])
            if n != s.n_new_sls - 1:
                p_avg0 = (s.p_grid[n, k] + s.p_grid[n, k + 1] + s.p_grid[n + 1, k]) / 3.0
                p_avg1 = (s.p_grid[n + 1, k + 1] + s.p_grid[n, k + 1] + s.p_grid[n + 1, k]) / 3.0
                P3 = (s.x_grid[n + 1, k],     s.y_grid[n + 1, k],     s.z_grid[n + 1, k])
                P4 = (s.x_grid[n + 1, k + 1], s.y_grid[n + 1, k + 1], s.z_grid[n + 1, k + 1])
            else:
                p_avg0 = (s.p_grid[n, k] + s.p_grid[n, k + 1] + s.p_grid[0, k]) / 3.0
                p_avg1 = (s.p_grid[0, k + 1] + s.p_grid[n, k + 1] + s.p_grid[0, k]) / 3.0
                P3 = (s.x_grid[0, k],     s.y_grid[0, k],     s.z_grid[0, k])
                P4 = (s.x_grid[0, k + 1], s.y_grid[0, k + 1], s.z_grid[0, k + 1])
            # Same literal-index pressure check as the X loop above.
            p_n1k_p  = s.p_grid[n + 1, k]
            p_n1k1_p = s.p_grid[n + 1, k + 1]
            rot = s.rotate_sl[n]
            tmp_sa, tmp_area, tmp_force = _proc(
                P1, P2, P3, P4, p_avg0, p_avg1,
                s.p_grid[n, k]     != 0.0,
                s.p_grid[n, k + 1] != 0.0,
                p_n1k_p            != 0.0,
                p_n1k1_p           != 0.0,
                rot, tmp_sa, tmp_area, tmp_force,
            )
        if f_avs_sl:
            f_avs_sl.write(f"{n}\t{_fmt(tmp_area + s.a_throat)}\t"
                           f"{_fmt(tmp_sa)}\t{_fmt(tmp_force)}\n")
    if f_avs_sl:
        f_avs_sl.close()


# ---------------------------------------------------------------------------
# FindMaxX                                                                   -
# ---------------------------------------------------------------------------
def find_max_x(state: "STTState") -> float:
    """Direct port of ``CSTT2001Dlg::FindMaxX``.

    Reads the MOC grid (already loaded as ``state.moc_grid``) and
    determines the largest valid X for the trimmed contour, by
    locating where a reflective wave would first interfere with the
    user's trimmed streamlines.
    """
    s = state
    if s.moc_grid is None:
        return -1.0

    grid = s.moc_grid
    min_trim_x = 9e9
    min_trim_r = 0.0
    min_trim_i = -1
    for n in range(s.n_new_sls):
        j = int(s.new_sl_pts[n])
        if s.xt[n, j] < min_trim_x:
            min_trim_x = s.xt[n, j]
            min_trim_r = math.sqrt(s.yt[n, j] ** 2 + s.zt[n, j] ** 2)
            min_trim_i = n
    if min_trim_i < 0:
        return -1.0

    min_dist = 9e9
    i_match = 0
    j_match = -99
    for jj in range(grid.n_zones):
        k = 0
        last = int(grid.last_pt[jj])
        while k <= last and grid.x[k, jj] < min_trim_x:
            k += 1
        if k <= last:
            dist = math.sqrt((grid.x[k, jj] - min_trim_x) ** 2
                             + (grid.r[k, jj] - min_trim_r) ** 2)
            if dist < min_dist:
                min_dist = dist
                i_match = int(grid.i[k, jj])
                j_match = jj
    if j_match < 0:
        return -1.0

    i_axial = int(grid.i[int(grid.last_pt[j_match]), j_match])
    if grid.r[i_axial, j_match] > 0.0:
        return -1.0

    for m in range(1, i_axial + 1):
        if (i_axial - m) < 0 or (j_match + m) >= grid.n_zones:
            break
        x_rw = grid.x[i_axial - m, j_match + m]
        r_rw = grid.r[i_axial - m, j_match + m]
        for n in range(s.n_new_sls):
            last_pt_n = int(s.new_sl_pts[n])
            i = 0
            while i <= last_pt_n and s.xt[n, i] < x_rw:
                i += 1
            if i <= last_pt_n:
                rad = math.sqrt(s.yt[n, i - 1] ** 2 + s.zt[n, i - 1] ** 2)
                if rad < r_rw:
                    return x_rw
    return -1.0


# ---------------------------------------------------------------------------
# CalcXStatus                                                                -
# ---------------------------------------------------------------------------
def calc_x_status(n: int, x: float, x1, y1) -> tuple[float, float]:
    """Direct port of ``CSTT2001Dlg::CalcXStatus``.

    Finds the Y value and local wall angle (deg) at axial position ``x``
    along the streamline given by arrays ``x1`` / ``y1``. Returns
    ``(y, theta_deg)``.
    """
    theta = 0.0
    i = 1
    for i in range(1, n):
        if x1[i] != x1[i - 1]:
            theta = math.atan((y1[i] - y1[i - 1])
                              / (x1[i] - x1[i - 1])) * DEG_PER_RAD
        if x1[i] > x:
            break
    return y1[i], theta
