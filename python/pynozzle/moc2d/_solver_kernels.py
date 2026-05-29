"""Numerical kernels for :class:`pynozzle.moc2d.solver.MOCGridCalc`.

This module is imported by :mod:`pynozzle.moc2d.solver` and attaches the
remaining methods to the class.  Splitting the file this way keeps each
piece browsable without abandoning the "one big class, like the C++"
structure that makes the code easier to compare against the original.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..common.constants import (
    AXI, CONE, DEG_PER_RAD, ENDPOINT, EPS, EXITMACH, EXITPRESSURE,
    FIXEDEND, GASCON, GRAV, NOZZLELENGTH, PERFECT, PI, RAD_PER_DEG, RAO,
    SEC_FAIL, SEC_FAIL_HIGH, SEC_FAIL_LOW, SEC_OK, TWOD,
)
from ..common.thermo import (
    calc_A, calc_B, calc_MM, calc_R, calc_R_star, calc_b, calc_mu,
    l_dy_dx, r_dy_dx, tan_avg,
)
from .solver import MOCGridCalc, _DEPoint, _HARD_FAIL, _SecantFail


# ---------------------------------------------------------------------------
# 1.  Initial throat line                                                     -
# ---------------------------------------------------------------------------
def _kl_throat(self: MOCGridCalc, i: int, geom: int, RS: float) -> None:
    """Direct port of ``MOC_GridCalc::KLThroat``.

    Sets ``theta[i,0]``, ``mach[i,0]`` and the isentropic state at the
    starting characteristic point ``i`` for the given upstream radius
    ``RS`` and geometry. Raises :class:`_SecantFail` if the local Mach
    drops below 1.0 (subsonic throat).
    """
    g = self.grid
    y = g.r[i, 0]
    G = g.gamma[i, 0]
    u = [0.0] * 4
    v = [0.0] * 4

    if geom == AXI:
        z = g.x[i, 0] * math.sqrt(2 * RS / (G + 1))
        RSP = RS + 1.0
        u[1] = y*y/2 - 0.25 + z
        v[1] = y*y*y/4 - y/4 + y*z
        u[2] = ((2*G+9)*y**4/24 - (4*G+15)*y*y/24 + (10*G+57)/288
                # The original C++ has ``y*y - 5/8`` here, where ``5/8`` is
                # integer division and so evaluates to 0. Python would
                # produce 0.625, which differs from the sample outputs;
                # we preserve the C++ behavior by writing it as ``y*y``.
                + z*(y*y) - (2*G-3)*z*z/6)
        v[2] = ((G+3)*y**5/9 - (20*G+63)*y**3/96 + (28*G+93)*y/288
                + z*((2*G+9)*y**3/6 - (4*G+15)*y/12) + y*z*z)
        u[3] = ((556*G*G + 1737*G + 3069)*y**6/10368
                - (388*G*G + 1161*G + 1881)*y**4/2304
                + (304*G*G + 831*G + 1242)*y*y/1728
                - (2708*G*G + 7839*G + 14211)/82944
                + z*((52*G*G + 51*G + 327)*y**4/34
                     - (52*G*G + 75*G + 279)*y*y/192
                     + (92*G*G + 180*G + 639)/1152)
                + z*z*(-(7*G - 3)*y*y/8 + (13*G - 27)/48)
                + (4*G*G - 57*G + 27)*z**3/144)
        # The next group has a transcription quirk in the original code
        # (the ``*`` in ``z*(... * (388*G*G + 1161*G + 1181)*y**3/576``
        # appears where ``-`` was almost certainly intended -- but to keep
        # the port byte-faithful we reproduce it exactly).
        v[3] = ((6836*G*G + 23031*G + 30627)*y**7/82944
                - (3380*G*G + 11391*G + 15291)*y**5/13824
                + (3424*G*G + 11271*G + 15228)*y**3/13824
                - (7100*G*G + 22311*G + 30249)*y/82944
                + z*((556*G*G + 1737*G + 3069)*y**5/1728
                     * (388*G*G + 1161*G + 1181)*y*y/576
                     + (304*G*G + 831*G + 1242)*y/864)
                + z*z*((52*G*G + 51*G + 327)*y**3/192
                       - (52*G*G + 75*G + 279)*y/192)
                - z**3*(7*G - 3)*y/12)
        U = (1 + u[1]/RSP + (u[1] + u[2])/(RSP*RSP)
             + (u[1] + 2*u[2] + u[3])/(RSP**3))
        V = (math.sqrt((G + 1) / (2 * RSP))
             * (v[1]/RSP + (1.5*v[1] + v[2])/(RSP*RSP)
                + (15./8.*v[1] + 2.5*v[2] + v[3])/(RSP**3)))
    elif geom == TWOD:
        z = g.x[i, 0] * math.sqrt(RS / (G + 1))
        # NOTE: the C++ source has ``- 1/6`` here which evaluates to 0 via
        # integer division. Python's ``1/6`` is 0.1667, so we write 0
        # explicitly to preserve the C++ behavior.
        u[1] = 0.5*y*y - 0 + z
        v[1] = y*y*y/6 - y/6 + y*z
        u[2] = ((y+6)*y**4/18 - (2*G+9)*y*y/18 + (G+30)/270
                + z*(y*y - 0.5) - (2*G-3)*z*z/6)
        v[2] = ((22*G+75)*y**5/360 - (5*G+21)*y**3/54 + (34*G+195)*y/1080
                + z/9*((2*G+12)*y**3 - (2*G+9)*y) + y*z*z)
        u[3] = ((362*G*G + 1449*G + 3177)*y**6/12960
                - (194*G*G + 837*G + 1665)*y**4/2592
                + (854*G*G + 3687*G + 6759)*y*y/12960
                - (782*G*G + 5523 + 2*G*2887)/272160
                + z*((26*G*G + 27*G + 237)*y**4/288
                     - (26*G*G + 51*G + 189)*y*y/144
                     + (134*G*G + 429*G + 1743)/4320)
                + z*z*(-5*G*y*y/4 + (7*G - 18)/36)
                + z**3*(2*G*G - 33*G + 9)/72)
        # Same transcription oddity as the AXI branch -- preserved verbatim.
        v[3] = ((6574*G*G + 26481*G + 40059)*y**7/181440
                - (2254*G*G + 10113*G + 16479)*y**5/25920
                + (5026*G*G + 25551*G + 46377)*y**3/77760
                - (7570*G*G + 45927*G + 98757)*y/544320
                + z*((362*G*G + 1449*G + 3177)*y**5/2160
                     * (194*G*G + 837*G + 1665)*y**3/648
                     + (854*G*G + 3687*G + 6759)*y/6480)
                + z*z*((26*G*G + 27*G + 237)*y**3/144
                       - (26*G*G + 51*G + 189)/144)
                + z**3*(-5*G*y/6))
        U = 1 + u[1]/RS + u[2]/RS/RS + u[3]/RS/RS/RS
        V = (math.sqrt((G + 1)/RS)
             * (v[1]/RS + v[2]/RS/RS + v[3]/RS/RS/RS))
    else:
        raise _SecantFail("Unknown geometry passed to KLThroat")

    if abs(V) < 1e-5:
        V = 0.0
    th = math.atan2(V, U)
    if abs(th) < 1e-5:
        th = 0.0
    g.theta[i, 0] = th
    Q = math.sqrt(U*U + V*V)
    g.mach[i, 0] = Q
    self._set_isentropic(i, 0, g.gamma[i, 0], g.mach[i, 0])
    if g.mach[i, 0] < 1.0:
        raise _SecantFail(
            "Initial data line is subsonic at i = "
            f"{i}; try increasing the initial-line angle.")


def _sauer(self: MOCGridCalc, i: int, geom: int, RS: float) -> None:
    """Direct port of ``MOC_GridCalc::Sauer``.

    Modified Sauer transonic for axisymmetric or planar flow (used as an
    alternative to KLThroat in the original; kept for completeness even
    though the C++ ``CalcInitialThroatLine`` currently uses KLThroat).
    """
    g = self.grid
    DELTA = 1.0
    R_ = g.r[i, 0]
    X_ = g.x[i, 0]
    G = g.gamma[i, 0]
    G1 = G + 1.0
    RRA = RS + DELTA * G / 4.0
    Z = X_ * math.sqrt(RS / RRA)
    if geom == AXI:
        B1 = math.sqrt(2.0 / (G1 * RRA))
        B0 = -1.0 / (4.0 * RRA)
        UP = (G1 * (B1 * R_)**2) / 4.0 + B0 + B1 * Z
        VP = ((G1*G1*(B1*R_)**3) / 16.0 + G1*B1*B0*R_/2.0
              + G1*B1*B1*R_*X_/2.0)
    elif geom == TWOD:
        B1 = math.sqrt(1.0 / (G1 * RRA))
        B0 = -1.0 / (6.0 * RRA)
        UP = (G1 * (B1 * R_)**2) / 2.0 + B0 + B1 * Z
        VP = ((G1*G1*(B1*R_)**3) / 6.0 + G1*B1*B0*R_
              + G1*B1*B1*R_*X_)
    else:
        raise _SecantFail("Unknown geometry passed to Sauer")
    U = 1.0 + UP
    V = VP
    th = math.atan2(V, U)
    if th < 0.0:
        th = 0.0
    g.theta[i, 0] = th
    QB = math.sqrt(U*U + V*V)
    g.mach[i, 0] = QB / math.sqrt((G1 - (G - 1.0)*QB*QB) / 2.0)
    self._set_isentropic(i, 0, g.gamma[i, 0], g.mach[i, 0])


def _calc_initial_throat_line(
    self: MOCGridCalc, r_up: float, n: int, gamma: float,
    p_amb: float, geom: int, t_flag: int, m_t: float,
) -> None:
    """Direct port of ``MOC_GridCalc::CalcInitialThroatLine``.

    Populates ``j = 0`` with the initial data line (T-T') via the
    Kliegel-Levine method (or directly from the user-given throat Mach
    number, when ``t_flag == 1``).
    """
    g = self.grid
    g.x[0, 0] = 0.0
    drdx = 0.0
    for i in range(n + 1):
        g.gamma[i, 0] = gamma
        g.r[i, 0] = math.pow(math.sin(1.5707963 * (n - i) / n), 1.5)
        if i != 0:
            drdx = self.rDyDx(g.theta[i-1, 0], self.CalcMu(g.mach[i-1, 0]))
        g.mach[i, 0] = 2.0
        if not t_flag:
            if i != 0:
                while g.mach[i, 0] > 1.5:
                    if i > 0:
                        g.x[i, 0] = g.x[i-1, 0] + (g.r[i, 0] - g.r[i-1, 0]) / drdx
                    _kl_throat(self, i, geom, r_up)
                    drdx *= 2.0
            else:
                _kl_throat(self, i, geom, r_up)
        else:
            g.theta[i, 0] = 0.0
            g.mach[i, 0] = m_t
            if i > 0:
                g.x[i, 0] = g.x[i-1, 0] + (g.r[i, 0] - g.r[i-1, 0]) / drdx
            if g.mach[i, 0] < 1.0:
                raise _SecantFail(
                    "Calculated throat Mach number < 1.0 at i = " + str(i))
            self._set_isentropic(i, 0, g.gamma[i, 0], g.mach[i, 0])

    g.i_last[0] = n
    _calc_massflow_and_thrust(self, 0, 0, p_amb, geom)


MOCGridCalc._sauer = _sauer
MOCGridCalc._kl_throat = _kl_throat
MOCGridCalc._calc_initial_throat_line = _calc_initial_throat_line


# ---------------------------------------------------------------------------
# 2.  Wall-point routines                                                     -
# ---------------------------------------------------------------------------
def _calc_arc_wall_point(
    self: MOCGridCalc, j: int, rad: float, beta_max: float,
    db_limit: float, geom: int,
) -> int:
    """Direct port of ``MOC_GridCalc::CalcArcWallPoint``.

    Returns 1 if a special wall point was inserted (so that the caller
    bumps ``iLast[j]``), 0 otherwise.
    """
    g = self.grid
    flag_special = 0
    slrc = [0.0] * 4
    A = [0.0] * 4
    B = [0.0] * 4
    R = [0.0] * 4

    slrc[1] = self.lDyDx(g.theta[1, j-1], self.CalcMu(g.mach[1, j-1]))
    A[1] = self.CalcA(g.mach[1, j-1], g.gamma[1, j-1])
    B[1] = self.CalcB(g.mach[1, j-1], g.theta[1, j-1], g.r[1, j-1])
    R[1] = self.CalcR(g.mach[1, j-1], g.theta[1, j-1], g.r[1, j-1])

    g.r[0, j] = g.r[0, j-1]
    g.x[0, j] = g.x[1, j-1]
    g.gamma[0, j] = g.gamma[1, j-1]
    slrc[3] = slrc[1]
    A[3] = A[1]
    B[3] = B[1]
    R[3] = R[1]

    x_err = r_err = m_err = t_err = 9.9
    x3_old = r3_old = m3_old = theta3_old = 9.9
    k = 0
    while ((abs(x_err) > self.con_crit or abs(r_err) > self.con_crit
            or abs(m_err) > self.con_crit or abs(t_err) > self.con_crit)
           and k < 50):
        k += 1
        slrc13 = self.TanAvg(slrc[1], slrc[3])
        g.x[0, j] = (g.r[0, j] - g.r[1, j-1]) / slrc13 + g.x[1, j-1]
        g.r[0, j] = 1 + rad - math.sqrt(rad*rad - g.x[0, j]*g.x[0, j])
        g.theta[0, j] = math.asin(g.x[0, j] / rad)

        if geom == AXI:
            if B[1] <= R[1]:
                T1 = (g.x[0, j] - g.x[1, j-1]) * (B[3] + B[1])
            else:
                T1 = (g.r[0, j] - g.r[1, j-1]) * (R[3] + R[1])
        else:
            T1 = 0.0

        g.mach[0, j] = (g.mach[1, j-1]
                        + (g.theta[0, j] - g.theta[1, j-1] + 0.5*T1)
                        / (0.5 * (A[1] + A[3])))

        slrc[3] = self.lDyDx(g.theta[0, j], self.CalcMu(g.mach[0, j]))
        A[3] = self.CalcA(g.mach[0, j], g.gamma[0, j])
        B[3] = self.CalcB(g.mach[0, j], g.theta[0, j], g.r[0, j])
        R[3] = self.CalcR(g.mach[0, j], g.theta[0, j], g.r[0, j])

        r_err = (g.r[0, j] - r3_old) / r3_old
        x_err = (g.x[0, j] - x3_old) / x3_old
        m_err = (g.mach[0, j] - m3_old) / m3_old
        t_err = (g.theta[0, j] - theta3_old) / theta3_old
        x3_old, r3_old = g.x[0, j], g.r[0, j]
        m3_old, theta3_old = g.mach[0, j], g.theta[0, j]
    if k >= 50:
        raise _SecantFail("Could not converge on arc wall point")

    if (g.theta[0, j] - g.theta[0, j-1]) > db_limit or g.theta[0, j] > beta_max:
        alpha = min(beta_max, g.theta[0, j-1] + 0.5 * db_limit)
        flag_special = 1
        _calc_special_wall_point(self, j, rad, alpha)
    self._set_isentropic(0, j, g.gamma[0, j], g.mach[0, j])
    return flag_special


def _calc_cone_wall_point(
    self: MOCGridCalc, j: int, cone_angle_rad: float, geom: int,
) -> None:
    """Direct port of ``MOC_GridCalc::CalcConeWallPoint``."""
    g = self.grid
    slrc = [0.0] * 4
    A = [0.0] * 4
    B = [0.0] * 4
    R = [0.0] * 4

    slrc[1] = self.lDyDx(g.theta[1, j-1], self.CalcMu(g.mach[1, j-1]))
    A[1] = self.CalcA(g.mach[1, j-1], g.gamma[1, j-1])
    B[1] = self.CalcB(g.mach[1, j-1], g.theta[1, j-1], g.r[1, j-1])
    R[1] = self.CalcR(g.mach[1, j-1], g.theta[1, j-1], g.r[1, j-1])

    g.x[0, j] = ((g.r[1, j-1] - g.r[0, j-1] - slrc[1]*g.x[1, j-1]
                  + math.tan(cone_angle_rad) * g.x[0, j-1])
                 / (math.tan(cone_angle_rad) - slrc[1]))
    g.r[0, j] = g.r[1, j-1] + slrc[1] * (g.x[0, j] - g.x[1, j-1])
    g.theta[0, j] = cone_angle_rad
    g.gamma[0, j] = g.gamma[1, j-1]

    A[3] = A[1]
    B[3] = B[1]
    R[3] = R[1]
    slrc[3] = self.lDyDx(g.theta[0, j], self.CalcMu(g.mach[1, j-1]))

    m_err = 9.9
    m3_old = 9.9
    k = 0
    while abs(m_err) > 1e-8 and k < 50:
        k += 1
        if geom == AXI:
            if B[1] <= R[1]:
                T1 = (g.x[0, j] - g.x[1, j-1]) * (B[3] + B[1])
            else:
                T1 = (g.r[0, j] - g.r[1, j-1]) * (R[3] + R[1])
        else:
            T1 = 0.0
        g.mach[0, j] = (g.mach[1, j-1]
                        + (g.theta[0, j] - g.theta[1, j-1] + 0.5*T1)
                        / (0.5 * (A[1] + A[3])))
        slrc[3] = self.lDyDx(g.theta[0, j], self.CalcMu(g.mach[0, j]))
        A[3] = self.CalcA(g.mach[0, j], g.gamma[0, j])
        B[3] = self.CalcB(g.mach[0, j], g.theta[0, j], g.r[0, j])
        R[3] = self.CalcR(g.mach[0, j], g.theta[0, j], g.r[0, j])
        m_err = (g.mach[0, j] - m3_old) / m3_old
        m3_old = g.mach[0, j]
    if k >= 50:
        raise _SecantFail("Could not converge on cone wall point")
    self._set_isentropic(0, j, g.gamma[0, j], g.mach[0, j])


def _calc_special_wall_point(
    self: MOCGridCalc, j: int, rad: float, alpha2: float,
) -> None:
    """Direct port of ``MOC_GridCalc::CalcSpecialWallPoint``.

    Inserts a wall point at a prescribed angle ``alpha2`` along the
    downstream throat arc whenever two successive arc-wall points exceed
    the user's ``dt_limit``.
    """
    g = self.grid
    slrc = [0.0] * 5
    A = [0.0] * 5
    B = [0.0] * 5
    R = [0.0] * 5
    srrc = [0.0] * 4

    g.theta[0, j] = alpha2
    g.x[0, j] = rad * math.sin(alpha2)
    g.r[0, j] = 1 + rad * (1 - math.cos(alpha2))
    g.gamma[0, j] = g.gamma[1, j-1]

    slrc[1] = self.lDyDx(g.theta[1, j-1], self.CalcMu(g.mach[1, j-1]))
    slrc[2] = self.lDyDx(g.theta[0, j-1], self.CalcMu(g.mach[0, j-1]))
    A[1] = self.CalcA(g.mach[1, j-1], g.gamma[1, j-1])
    B[1] = self.CalcB(g.mach[1, j-1], g.theta[1, j-1], g.r[1, j-1])
    R[1] = self.CalcR(g.mach[1, j-1], g.theta[1, j-1], g.r[1, j-1])
    srrc[1] = self.rDyDx(g.theta[1, j-1], self.CalcMu(g.mach[1, j-1]))
    srrc[2] = self.rDyDx(g.theta[0, j-1], self.CalcMu(g.mach[0, j-1]))
    A[2] = self.CalcA(g.mach[0, j-1], g.gamma[0, j-1])
    B[2] = self.CalcB(g.mach[0, j-1], g.theta[0, j-1], g.r[0, j-1])
    R[2] = self.CalcR(g.mach[0, j-1], g.theta[0, j-1], g.r[0, j-1])

    slrc[3] = slrc[1]
    slrc[4] = slrc[2]
    A[3] = A[1]
    B[3] = B[1]
    R[3] = R[1]
    s4rrc = self.TanAvg(srrc[1], srrc[2])

    A_err = B_err = R_err = K_err = 9.9
    k = 0
    while ((abs(A_err) > self.con_crit or abs(B_err) > self.con_crit
            or abs(R_err) > self.con_crit or abs(K_err) > self.con_crit)
           and k < 50):
        k += 1
        slope34 = self.TanAvg(slrc[3], slrc[4])
        if abs(slope34) < 10000:
            x4 = ((g.r[0, j] - g.r[0, j-1] + s4rrc*g.x[0, j-1]
                   - slope34*g.x[0, j])
                  / (s4rrc - slope34))
        else:
            x4 = g.x[0, j]
        ratio = (x4 - g.x[1, j-1]) / (g.x[0, j-1] - g.x[1, j-1])
        A4 = A[1] + ratio * (A[2] - A[1])
        theta4 = g.theta[1, j-1] + ratio * (g.theta[0, j-1] - g.theta[1, j-1])
        slrc[4] = slrc[1] + ratio * (slrc[2] - slrc[1])
        M4 = g.mach[1, j-1] + ratio * (g.mach[0, j-1] - g.mach[1, j-1])

        if abs(B[2]) <= abs(R[2]):
            B4 = B[1] + ratio*(B[2] - B[1])
            T4 = (g.x[0, j] - x4) * (B[3] + B4)
        else:
            R4 = R[1] + ratio*(R[2] - R[1])
            r4 = g.r[1, j-1] + ratio*(g.r[0, j-1] - g.r[1, j-1])
            T4 = (g.r[0, j] - r4) * (R[3] + R4)

        g.mach[0, j] = M4 + (g.theta[0, j] - theta4 + 0.5*T4) / (0.5*(A4 + A[3]))

        K_new = self.lDyDx(g.theta[0, j], self.CalcMu(g.mach[0, j]))
        A_new = self.CalcA(g.mach[0, j], g.gamma[0, j])
        B_new = self.CalcB(g.mach[0, j], g.theta[0, j], g.r[0, j])
        R_new = self.CalcR(g.mach[0, j], g.theta[0, j], g.r[0, j])
        K_err = (K_new - slrc[3]) / slrc[3] if slrc[3] != 0 else 0.0
        A_err = (A_new - A[3]) / A[3] if A[3] != 0 else 0.0
        B_err = (B_new - B[3]) / B[3] if B[3] != 0 else 0.0
        R_err = (R_new - R[3]) / R[3] if R[3] != 0 else 0.0
        slrc[3], A[3], B[3], R[3] = K_new, A_new, B_new, R_new
    self._set_isentropic(0, j, g.gamma[0, j], g.mach[0, j])


def _calc_contour_wall_point(self: MOCGridCalc, j: int, i_bottom: int) -> None:
    """Direct port of ``MOC_GridCalc::CalcContourWallPoint`` (used by 2D
    nozzle to set the lower wall once the upper wall point is known).
    """
    g = self.grid
    g.mach[0, j]  = g.mach[1, j-1]
    g.pres[0, j]  = g.pres[1, j-1]
    g.temp[0, j]  = g.temp[1, j-1]
    g.gamma[0, j] = g.gamma[1, j-1]
    g.rho[0, j]   = g.rho[1, j-1]
    g.theta[0, j] = g.theta[1, j-1]

    slope1 = math.tan(0.5 * (g.theta[0, j] + g.theta[0, j-1]))
    mu2 = math.asin(1.0 / g.mach[1, j-1])
    slope2 = math.tan(g.theta[1, j-1] + mu2)
    g.x[0, j] = ((g.r[0, j-1] - g.r[1, j-1]
                  - g.x[0, j-1]*slope1 + g.x[1, j-1]*slope2)
                 / (slope2 - slope1))
    g.r[0, j] = slope1 * (g.x[0, j] - g.x[0, j-1]) + g.r[0, j-1]
    # mirror to lower wall
    g.r[i_bottom, j]     = -g.r[0, j]
    g.x[i_bottom, j]     = g.x[0, j]
    g.mach[i_bottom, j]  = g.mach[0, j]
    g.pres[i_bottom, j]  = g.pres[0, j]
    g.temp[i_bottom, j]  = g.temp[0, j]
    g.rho[i_bottom, j]   = g.rho[0, j]
    g.gamma[i_bottom, j] = g.gamma[0, j]
    g.theta[i_bottom, j] = -g.theta[0, j]


MOCGridCalc._calc_arc_wall_point      = _calc_arc_wall_point
MOCGridCalc._calc_cone_wall_point     = _calc_cone_wall_point
MOCGridCalc._calc_special_wall_point  = _calc_special_wall_point
MOCGridCalc._calc_contour_wall_point  = _calc_contour_wall_point


# ---------------------------------------------------------------------------
# 3.  Interior / axial mesh-point solvers                                     -
# ---------------------------------------------------------------------------
def _calc_interior_mesh_points(
    self: MOCGridCalc, j: int, i_start: int, i_end: int,
    flag: int, geom: int,
) -> bool:
    """Direct port of ``MOC_GridCalc::CalcInteriorMeshPoints``.

    For each interior point (i, j), solves the LRC/RRC compatibility
    equations using RAO eq. 15, iterating the slopes to convergence.
    Returns True on success, False on hard failure.
    """
    g = self.grid

    for i in range(i_start, i_end):
        min_m_err = 9.9
        min_err_mach = min_err_theta = 0.0
        min_err_x = min_err_r = 0.0
        min_x_err = min_r_err = 0.0
        x_err = r_err = m_err = 9.9
        x3_old = r3_old = m3_old = 9.9
        M3 = 9.9
        k = 0
        ii = i if flag else i + 1
        if ii > g.i_last[j-1]:
            ii = int(g.i_last[j-1])

        s1 = self.lDyDx(g.theta[ii, j-1], self.CalcMu(g.mach[ii, j-1]))
        A1 = self.CalcA(g.mach[ii, j-1], g.gamma[ii, j-1])
        M1 = g.mach[ii, j-1]
        TH1 = g.theta[ii, j-1]
        if g.r[ii, j-1] != 0.0:
            B1 = self.CalcB(g.mach[ii, j-1], g.theta[ii, j-1], g.r[ii, j-1])
            R1 = self.CalcR(g.mach[ii, j-1], g.theta[ii, j-1], g.r[ii, j-1])
        else:
            if geom == TWOD:
                R1 = 0.0
                B1 = 0.0
            else:
                R1 = self.CalcR(g.mach[ii-1, j-1], g.theta[ii-1, j-1], g.r[ii-1, j-1])
                B1 = self.CalcB(g.mach[ii-1, j-1], g.theta[ii-1, j-1], g.r[ii-1, j-1])

        s2 = self.rDyDx(g.theta[i-1, j], self.CalcMu(g.mach[i-1, j]))
        A2 = self.CalcA(g.mach[i-1, j], g.gamma[i-1, j])
        M2 = g.mach[i-1, j]
        TH2 = g.theta[i-1, j]
        B2 = self.CalcB(g.mach[i-1, j], g.theta[i-1, j], g.r[i-1, j])
        b2 = self.Calcb(g.mach[i-1, j], g.theta[i-1, j], g.r[i-1, j])
        R2 = self.CalcR(g.mach[i-1, j], g.theta[i-1, j], g.r[i-1, j])
        RS2 = self.CalcRStar(g.mach[i-1, j], g.theta[i-1, j], g.r[i-1, j])

        # point-3 initial guesses
        s3lrc = s1
        s3rrc = s2
        b3 = b2
        B3 = B1
        R3 = R1
        RS3 = RS2
        A3 = (A1 + A2) / 2.0
        G3 = 0.5 * (g.gamma[ii, j-1] + g.gamma[i-1, j])
        TH3 = TH1

        while ((abs(x_err) > self.con_crit or abs(r_err) > self.con_crit
                or abs(m_err) > self.con_crit) and k < 1000 and M3 >= 1.0):
            k += 1
            slope13 = self.TanAvg(s1, s3lrc)
            slope23 = self.TanAvg(s2, s3rrc)

            if slope13 > 10000.0:
                g.x[i, j] = g.x[i-1, j]
            elif slope23 > 10000.0:
                g.x[i, j] = g.x[ii, j-1]
            else:
                g.x[i, j] = ((g.r[ii, j-1] - g.r[i-1, j]
                              - slope13*g.x[ii, j-1] + slope23*g.x[i-1, j])
                             / (slope23 - slope13))

            if abs(s2) <= abs(s1):
                g.r[i, j] = g.r[i-1, j] + slope23 * (g.x[i, j] - g.x[i-1, j])
            else:
                g.r[i, j] = g.r[ii, j-1] + slope13 * (g.x[i, j] - g.x[ii, j-1])

            if geom == TWOD:
                T1 = T2 = 0.0
            else:
                if abs(b2) <= abs(RS2):
                    T2 = (g.x[i, j] - g.x[i-1, j]) * (b2 + b3)
                else:
                    T2 = (g.r[i, j] - g.r[i-1, j]) * (RS3 + RS2)
                if abs(B1) <= abs(R1):
                    T1 = (g.x[i, j] - g.x[ii, j-1]) * (B1 + B3)
                else:
                    T1 = (g.r[i, j] - g.r[ii, j-1]) * (R3 + R1)

            M3 = ((2*(TH2 - TH1) + M2*(A2 + A3) + M1*(A1 + A3) + T1 + T2)
                  / (A1 + A2 + 2*A3))
            A3 = self.CalcA(M3, G3)
            TH3 = ((TH1 + TH2)/2.0
                   + 0.25*(M2*(A3 + A2) - M1*(A1 + A3)
                           - M3*(A2 - A1) + T2 - T1))
            if TH3 < 0.0:
                TH3 = 0.0
            s3lrc = self.lDyDx(TH3, self.CalcMu(M3))
            s3rrc = self.rDyDx(TH3, self.CalcMu(M3))
            B3 = self.CalcB(M3, TH3, g.r[i, j])
            b3 = self.Calcb(M3, TH3, g.r[i, j])
            R3 = self.CalcR(M3, TH3, g.r[i, j])
            RS3 = self.CalcRStar(M3, TH3, g.r[i, j])

            x_err = (g.x[i, j] - x3_old) / x3_old if x3_old != 0 else 0.0
            r_dif = g.r[i, j] - r3_old
            if abs(r_dif) < 1e-5 and abs(g.r[i, j]) < 1e-4:
                r_err = 1e-3 * r_dif
            else:
                r_err = (g.r[i, j] - r3_old) / r3_old if r3_old != 0 else 0.0
            m_err = (M3 - m3_old) / m3_old if m3_old != 0 else 0.0

            if abs(m_err) < min_m_err:
                min_m_err = abs(m_err)
                min_err_mach = M3
                min_err_theta = TH3
                min_err_x = g.x[i, j]
                min_err_r = g.r[i, j]
                min_x_err = x_err
                min_r_err = r_err
            x3_old = g.x[i, j]
            r3_old = g.r[i, j]
            m3_old = M3

        g.mach[i, j] = M3
        g.theta[i, j] = TH3
        g.gamma[i, j] = G3

        if k >= 1000 or M3 < 1.0:
            if min_m_err <= 5e-4:
                g.x[i, j] = min_err_x
                g.r[i, j] = min_err_r
                g.mach[i, j] = min_err_mach
                g.theta[i, j] = min_err_theta
            else:
                return False

        if g.r[i, j] < 0.0:
            g.i_last[j] = i
            break
        if g.theta[i-1, j] != 0.0:
            g.theta[i, j] = max(g.theta[i, j], 0.0)
        self._set_isentropic(i, j, g.gamma[i, j], g.mach[i, j])
    return True


def _calc_axial_mesh_point(self: MOCGridCalc, j: int, i_end: int) -> None:
    """Direct port of ``MOC_GridCalc::CalcAxialMeshPoint``."""
    g = self.grid
    g.r[i_end, j] = 0.0
    g.theta[i_end, j] = 0.0
    g.gamma[i_end, j] = g.gamma[i_end-1, j]

    s2 = self.rDyDx(g.theta[i_end-1, j], self.CalcMu(g.mach[i_end-1, j]))
    A2 = self.CalcA(g.mach[i_end-1, j], g.gamma[i_end-1, j])
    b2 = self.Calcb(g.mach[i_end-1, j], g.theta[i_end-1, j], g.r[i_end-1, j])
    s3 = s2
    A3 = A2
    m_err = x_err = 9.9
    x3_old = m3_old = 9.9
    k = 0
    while ((abs(m_err) > self.con_crit or abs(x_err) > self.con_crit)
           and k < 500):
        k += 1
        slope23 = self.TanAvg(s2, s3)
        g.x[i_end, j] = g.x[i_end-1, j] - g.r[i_end-1, j] / slope23
        g.mach[i_end, j] = (g.mach[i_end-1, j]
                            + 2*(g.theta[i_end-1, j]
                                 + b2*(g.x[i_end, j] - g.x[i_end-1, j]))
                            / (A2 + A3))
        s3 = self.rDyDx(g.theta[i_end, j], self.CalcMu(g.mach[i_end, j]))
        A3 = self.CalcA(g.mach[i_end, j], g.gamma[i_end, j])
        m_err = (g.mach[i_end, j] - m3_old) / m3_old if m3_old != 0 else 0.0
        x_err = (g.x[i_end, j] - x3_old) / x3_old if x3_old != 0 else 0.0
        x3_old = g.x[i_end, j]
        m3_old = g.mach[i_end, j]
    if k >= 500:
        raise _SecantFail("Could not find an axial mesh point")
    self._set_isentropic(i_end, j, g.gamma[i_end, j], g.mach[i_end, j])
    g.massflow[i_end, j] = 0.0
    g.thrust[i_end, j] = 0.0
    g.s_thrust[i_end, j] = 0.0


MOCGridCalc._calc_interior_mesh_points = _calc_interior_mesh_points
MOCGridCalc._calc_axial_mesh_point     = _calc_axial_mesh_point


# ---------------------------------------------------------------------------
# 4.  Mass-flow and thrust integration                                        -
# ---------------------------------------------------------------------------
def _calc_massflow_and_thrust(
    self: MOCGridCalc, j_start: int, j_end: int, p_amb: float, geom: int,
) -> None:
    """Direct port of ``MOC_GridCalc::CalcMassFlowAndThrustAlongMesh``.

    Integrates mass flow and thrust along each RRC from centerline to wall.
    """
    g = self.grid
    mol_wt = self.mol_wt
    for j in range(j_start, j_end + 1):
        n_axis = int(g.i_last[j])
        g.massflow[n_axis, j] = 0.0
        g.thrust[n_axis, j] = 0.0
        g.s_thrust[n_axis, j] = 0.0
        for i in range(n_axis - 1, -1, -1):
            a1 = math.sqrt(g.gamma[i, j] * GASCON / mol_wt * GRAV * g.temp[i, j])
            u1 = g.mach[i, j] * math.cos(g.theta[i, j]) * a1
            v1 = g.mach[i, j] * math.sin(g.theta[i, j]) * a1
            a2 = math.sqrt(g.gamma[i+1, j] * GASCON / mol_wt * GRAV * g.temp[i+1, j])
            u2 = g.mach[i+1, j] * math.cos(g.theta[i+1, j]) * a2
            v2 = g.mach[i+1, j] * math.sin(g.theta[i+1, j]) * a2

            dxdr = (g.x[i+1, j] - g.x[i, j]) / (g.r[i, j] - g.r[i+1, j])
            mdot_a = abs(0.5 * GRAV
                         * (g.rho[i, j]*u1 + g.rho[i+1, j]*u2
                            + dxdr*(g.rho[i, j]*v1 + g.rho[i+1, j]*v2)))
            f_a = abs((g.pres[i, j] - p_amb) * 144.0
                      + 0.5 * (g.rho[i, j]*u1*u1 + g.rho[i+1, j]*u2*u2
                               + dxdr*(g.rho[i, j]*u1*v1 + g.rho[i+1, j]*u2*v2)))
            Sf_a = abs(g.pres[i, j] * 144.0
                       + 0.5 * (g.rho[i, j]*u1*u1 + g.rho[i+1, j]*u2*u2
                                + dxdr*(g.rho[i, j]*u1*v1 + g.rho[i+1, j]*u2*v2)))
            if geom == TWOD:
                da = abs((g.r[i, j] - g.r[i+1, j]) / 12.0)
            else:
                da = abs(PI * (g.r[i, j]**2 - g.r[i+1, j]**2) / 144.0)
            g.massflow[i, j] = g.massflow[i+1, j] + mdot_a * da
            g.thrust[i, j]   = g.thrust[i+1, j]   + f_a    * da
            g.s_thrust[i, j] = g.s_thrust[i+1, j] + Sf_a   * da


MOCGridCalc._calc_massflow_and_thrust = _calc_massflow_and_thrust


# ---------------------------------------------------------------------------
# 5.  RRC-walking routines                                                    -
# ---------------------------------------------------------------------------
def _check_rrc_for_negative_points(self: MOCGridCalc, j: int) -> bool:
    g = self.grid
    for i in range(int(g.i_last[j]) + 1):
        if g.r[i, j] < 0.0:
            return True
    return False


def _calc_rrcs_along_arc(
    self: MOCGridCalc, j: int, rad: float, alpha_max: float,
    da_limit: float, p_amb: float, geom: int,
) -> int:
    """Direct port of ``MOC_GridCalc::CalcRRCsAlongArc``.

    Returns the index of the last computed RRC, or
    ``-(last good RRC)`` if the calculated axial Mach blew past 50 (signal
    for the caller to drop thetaB), or :data:`_HARD_FAIL` for a hard error.
    """
    g = self.grid
    x_arc_max = math.sin(alpha_max) * rad

    while abs(g.x[0, j] - x_arc_max) > 1e-6:
        j += 1
        g.massflow[0, j] = 10 * g.massflow[0, 0]
        special_flag = _calc_arc_wall_point(self, j, rad, alpha_max, da_limit, geom)
        if g.r[0, j] - g.r[0, j-1] <= 0.0:
            return _HARD_FAIL
        g.i_last[j] = g.i_last[j-1] + special_flag
        if not _calc_interior_mesh_points(self, j, 1, int(g.i_last[j]),
                                          special_flag, geom):
            return SEC_FAIL
        _calc_axial_mesh_point(self, j, int(g.i_last[j]))
        if g.mach[g.i_last[j], j] > 50.0:
            return -(j - 1)
        _calc_massflow_and_thrust(self, j, j, p_amb, geom)
        mdot_err = (g.massflow[0, j] - g.massflow[0, 0]) / g.massflow[0, 0]
        if abs(mdot_err) > 0.02:
            return _HARD_FAIL
        if _check_rrc_for_negative_points(self, j) and j != 1:
            for k in range(int(g.i_last[j]) + 1):
                g.x[k, j-1]        = g.x[k, j]
                g.r[k, j-1]        = g.r[k, j]
                g.mach[k, j-1]     = g.mach[k, j]
                g.theta[k, j-1]    = g.theta[k, j]
                g.pres[k, j-1]     = g.pres[k, j]
                g.temp[k, j-1]     = g.temp[k, j]
                g.rho[k, j-1]      = g.rho[k, j]
                g.gamma[k, j-1]    = g.gamma[k, j]
                g.i_last[j-1]      = g.i_last[j]
            j -= 1
    return j


MOCGridCalc._check_rrc_for_negative_points = _check_rrc_for_negative_points
MOCGridCalc._calc_rrcs_along_arc = _calc_rrcs_along_arc


# ---------------------------------------------------------------------------
# 6.  Cone-nozzle solver                                                     -
# ---------------------------------------------------------------------------
def _calc_cone_nozzle(
    self: MOCGridCalc, param_type: int, param_match: list[float],
    p_amb: float, geom: int, n_sl_i: int, n_sl_j: int,
) -> bool:
    """Direct port of ``MOC_GridCalc::CalcConeNozzle``.

    The "cone" case is simpler than the contoured case: the wall geometry
    is prescribed (a straight line at ``param_match[1]`` degrees beyond the
    downstream arc), so the only thing to do is to march along the cone
    until the user's exit constraint is met.
    """
    g = self.grid
    param_exit = param_match[0]
    cone_angle_rad = param_match[1] * RAD_PER_DEG
    j = _calc_rrcs_along_arc(self, 0, self.RWTD, cone_angle_rad,
                             self.DTLIMIT, p_amb, geom)
    if j == _HARD_FAIL:
        return False

    param_err = -1.0
    ratio = 0.0
    while param_err < 0.0:
        j += 1
        _calc_cone_wall_point(self, j, cone_angle_rad, geom)
        if param_type == EXITMACH:
            param_exit = g.mach[0, j]
        elif param_type == NOZZLELENGTH:
            param_exit = g.x[0, j]
        elif param_type == EXITPRESSURE:
            param_exit = self.p_total / g.pres[0, j]
        elif param_type == EPS:
            if geom == TWOD:
                param_exit = g.r[0, j]
            else:
                param_exit = g.r[0, j] * g.r[0, j]
        param_err = param_exit - param_match[0]
        if param_err < 0.0:
            g.i_last[j] = g.i_last[j-1]
            if not _calc_interior_mesh_points(self, j, 1, int(g.i_last[j]),
                                              0, geom):
                return False
            _calc_axial_mesh_point(self, j, int(g.i_last[j]))
            _calc_massflow_and_thrust(self, j, j, p_amb, geom)

    # Interpolate the wall end point to the requested exit parameter.
    if param_type == EXITMACH:
        ratio = (g.mach[0, j] - param_match[0]) / (g.mach[0, j] - g.mach[0, j-1])
    elif param_type == NOZZLELENGTH:
        ratio = (g.x[0, j] - param_match[0]) / (g.x[0, j] - g.x[0, j-1])
    elif param_type == EXITPRESSURE:
        ratio = ((self.p_total/g.pres[0, j] - param_match[0])
                 / (self.p_total/g.pres[0, j] - self.p_total/g.pres[0, j-1]))
    elif param_type == EPS:
        if geom == TWOD:
            ratio = (g.r[0, j] - param_match[0]) / (g.r[0, j] - g.r[0, j-1])
        else:
            ratio = ((g.r[0, j]**2 - param_match[0])
                     / (g.r[0, j]**2 - g.r[0, j-1]**2))

    for arr in (g.x, g.r, g.mach, g.pres, g.temp, g.theta, g.rho, g.gamma):
        arr[0, j] = arr[0, j] - ratio * (arr[0, j] - arr[0, j-1])

    self.theta_b_ans = cone_angle_rad
    g.i_last[j] = g.i_last[j-1]
    if not _calc_interior_mesh_points(self, j, 1, int(g.i_last[j]), 0, geom):
        return False
    _calc_axial_mesh_point(self, j, int(g.i_last[j]))
    _calc_massflow_and_thrust(self, j, j, p_amb, geom)

    self.last_rrc = j
    _crop_nozzle_to_length(self, self.last_rrc)
    # cone nozzles don't have a BD region
    self.j_bd = 0
    self.i_bd = 0
    self.j_de_last = 0
    self.mdot_err_ratio = (g.massflow[0, self.last_rrc]
                           / g.massflow[0, 0]) if g.massflow[0, 0] else 1.0
    return True


MOCGridCalc._calc_cone_nozzle = _calc_cone_nozzle


# ---------------------------------------------------------------------------
# 7.  Helpers used by the contoured-nozzle solver                            -
# ---------------------------------------------------------------------------
def _set_theta_b(self: MOCGridCalc, p_type: int, err: float, theta_b0: float) -> float:
    """Direct port of ``MOC_GridCalc::SetThetaB``."""
    if p_type != ENDPOINT:
        if err <= 0.0:
            self.theta_b_min = theta_b0
            return 1.2 * theta_b0
        self.theta_b_max = theta_b0
        return 0.8 * theta_b0
    else:
        if err >= 0.0:
            self.theta_b_min = theta_b0
            return 1.2 * theta_b0
        self.theta_b_max = theta_b0
        return 0.8 * theta_b0


MOCGridCalc._set_theta_b = _set_theta_b


def _crop_nozzle_to_length(self: MOCGridCalc, j_end: int) -> None:
    """Direct port of ``MOC_GridCalc::CropNozzleToLength``."""
    g = self.grid
    g.i_last[j_end] = 0
    for j in range(1, j_end):
        if g.x[g.i_last[j], j] >= g.x[0, j_end]:
            if g.r[g.i_last[j-1], j-1] == 0.0:
                new_last = int(g.i_last[j-1]) + 1
                g.r[new_last, j-1] = 0.0
                ratio = ((g.x[0, j_end] - g.x[g.i_last[j-1], j-1])
                         / (g.x[g.i_last[j], j] - g.x[g.i_last[j-1], j-1]))
                for arr in (g.x, g.mach, g.temp, g.pres, g.theta, g.rho, g.gamma):
                    arr[new_last, j-1] = (
                        arr[g.i_last[j-1], j-1]
                        + ratio * (arr[g.i_last[j], j] - arr[g.i_last[j-1], j-1])
                    )
                g.massflow[new_last, j-1] = 0.0
                g.thrust[new_last, j-1] = 0.0
                g.s_thrust[new_last, j-1] = 0.0
                g.i_last[j-1] = new_last

            i = 0
            while g.x[i, j] < g.x[0, j_end] and i <= g.i_last[j]:
                i += 1
            if i == 0:
                # copy from j-1
                for arr in (g.x, g.mach, g.theta, g.gamma,
                            g.rho, g.r, g.massflow, g.thrust, g.s_thrust):
                    arr[i, j] = arr[i, j-1]
                self._set_isentropic(i, j, g.gamma[i, j], g.mach[i, j-1])
                g.i_last[j] = 0
            else:
                dx_err = abs(g.x[i-1, j] - g.x[0, j_end]) / g.x[0, j_end]
                if dx_err < 1e-8:
                    g.i_last[j] = i - 1
                else:
                    if g.x[i, j] == g.x[i-1, j]:
                        ratio = 0.0
                    else:
                        ratio = ((g.x[0, j_end] - g.x[i-1, j])
                                 / (g.x[i, j] - g.x[i-1, j]))
                    g.x[i, j] = g.x[0, j_end]
                    for arr in (g.mach, g.theta, g.rho, g.r, g.gamma,
                                g.massflow, g.thrust, g.s_thrust):
                        arr[i, j] = arr[i-1, j] + ratio * (arr[i, j] - arr[i-1, j])
                    self._set_isentropic(i, j, g.gamma[i-1, j], g.mach[i, j])
                    g.i_last[j] = i


MOCGridCalc._crop_nozzle_to_length = _crop_nozzle_to_length


# ---------------------------------------------------------------------------
# 8.  Nozzle surface area helpers                                            -
# ---------------------------------------------------------------------------
def _calc_nozzle_surface_area(self: MOCGridCalc, j_last: int, geom: int) -> float:
    """Direct port of ``MOC_GridCalc::CalcNozzleSurfaceArea``."""
    g = self.grid
    sa = 0.0
    for j in range(1, j_last + 1):
        r_avg = 0.5 * (g.r[0, j] + g.r[0, j-1])
        length = math.sqrt((g.r[0, j] - g.r[0, j-1])**2
                           + (g.x[0, j] - g.x[0, j-1])**2)
        if geom == TWOD:
            sa += length * 12.0
        else:
            sa += length * 2.0 * PI * r_avg
    return sa


MOCGridCalc._calc_nozzle_surface_area = _calc_nozzle_surface_area
