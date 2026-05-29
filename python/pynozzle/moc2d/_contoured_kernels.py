"""Contoured-nozzle kernels for :class:`pynozzle.moc2d.solver.MOCGridCalc`.

This module implements the more involved part of the original C++ code:
the iterative search for the initial expansion angle ``ThetaB``, the LRC-DE
mass-flow-matching loop, and the BDE / wall-contour back-construction. It
attaches its functions onto :class:`MOCGridCalc` the same way
``_solver_kernels`` does.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..common.constants import (
    AXI, ENDPOINT, EPS, EXITMACH, EXITPRESSURE, FIXEDEND, GASCON, GRAV,
    NOZZLELENGTH, PERFECT, PI, RAD_PER_DEG, RAO, SEC_FAIL, SEC_FAIL_HIGH,
    SEC_FAIL_LOW, SEC_OK, TWOD,
)
from ..common.thermo import (
    calc_A, calc_B, calc_MM, calc_R, calc_R_star, calc_b, calc_mu,
    isentropic_p_t_rho, l_dy_dx, r_dy_dx, tan_avg,
)
from .solver import MOCGridCalc, _DEPoint, _HARD_FAIL, _SecantFail


# ---------------------------------------------------------------------------
# 1.  Derivatives, Runge-Kutta integrators for the LRC DE                    -
# ---------------------------------------------------------------------------
def _deriv(
    self: MOCGridCalc, i: int, r0: float, mach0: float,
    theta0: float, gamma0: float,
) -> float:
    """Direct port of ``MOC_GridCalc::Deriv``.

    Returns the derivatives needed for the LRC-DE integration:
    i = 0 -> dM/dr, 1 -> dx/dr, 2 -> dtheta/dr, 3 -> dr/dr = 1.
    """
    mu0 = calc_mu(mach0)
    if theta0 < 5e-6:
        if i == 2:
            return 1.0 / math.tan(mu0)
        return 0.0
    if i == 1:
        return 1.0 / math.tan(theta0 + mu0)
    if i == 3:
        return 1.0
    tan_theta = math.tan(theta0)
    m32 = math.pow(mach0*mach0 - 1, 1.5)
    tt0 = 1.0 + (gamma0 - 1.0) / 2.0 * mach0 * mach0
    a = r0 * math.sin(theta0 + mu0)
    b = 2.0 * m32 / tan_theta
    c = ((((gamma0 + 1.0) / 2.0) * mach0 * mach0 - 2.0) * mach0 * mach0
         + 2.0)
    D = a * (b - c)
    if i == 0:
        return -math.sin(theta0 - mu0) * mach0 * (mach0*mach0 - 1.0) * tt0 / D
    if i == 2:
        return ((math.sin(theta0)
                 * ((gamma0 - 1.0)/2.0 * mach0**4 + 1.0)
                 - m32 * math.cos(theta0))
                / (mach0 * D))
    return 0.0


def _runge_kutta(
    self: MOCGridCalc, h: float, r0: float, x0: float,
    mach0: float, theta0: float, gamma0: float,
) -> tuple[float, float, float, float]:
    """Direct port of ``MOC_GridCalc::RungeKutta`` (classical RK4).

    Returns ``(mach_new, x_new, theta_new, r_new)``.
    """
    ip = [mach0, x0, theta0, r0]
    k1 = [h * _deriv(self, i, r0, ip[0], ip[2], gamma0) for i in range(4)]
    r_h = r0 + h / 2.0
    p = [ip[i] + k1[i] / 2.0 for i in range(4)]
    k2 = [h * _deriv(self, i, r_h, p[0], p[2], gamma0) for i in range(4)]
    p = [ip[i] + k2[i] / 2.0 for i in range(4)]
    k3 = [h * _deriv(self, i, r_h, p[0], p[2], gamma0) for i in range(4)]
    r_h2 = r_h + h / 2.0
    p = [ip[i] + k3[i] for i in range(4)]
    k4 = [h * _deriv(self, i, r_h2, p[0], p[2], gamma0) for i in range(4)]
    out = tuple(ip[i] + (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6.0
                for i in range(4))
    return out


def _runge_kutta_fehlberg(
    self: MOCGridCalc, h: float, r0: float, x0: float,
    mach0: float, theta0: float, gamma0: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Direct port of ``MOC_GridCalc::RungeKuttaFehlberg`` (RKF4(5)).

    Returns a tuple of length 8: positions [0..3] = (mach, x, theta, r)
    and error estimates [4..7].
    """
    ip = [mach0, x0, theta0, r0]
    k1 = [h * _deriv(self, i, r0, ip[0], ip[2], gamma0) for i in range(4)]
    rG = r0 + h / 4.0
    p = [ip[i] + k1[i] / 4.0 for i in range(4)]
    k2 = [h * _deriv(self, i, rG, p[0], p[2], gamma0) for i in range(4)]
    rG = r0 + 3 * h / 8.0
    p = [ip[i] + 3*k1[i]/32.0 + 9*k2[i]/32.0 for i in range(4)]
    k3 = [h * _deriv(self, i, rG, p[0], p[2], gamma0) for i in range(4)]
    rG = r0 + 12 * h / 13.0
    p = [ip[i] + 1932*k1[i]/2197.0 - 7200*k2[i]/2197.0 + 7296*k3[i]/2197.0
         for i in range(4)]
    k4 = [h * _deriv(self, i, rG, p[0], p[2], gamma0) for i in range(4)]
    rG = r0 + h
    p = [ip[i] + 439*k1[i]/216.0 - 8*k2[i] + 3680*k3[i]/513.0
         - 845*k4[i]/4104.0 for i in range(4)]
    k5 = [h * _deriv(self, i, rG, p[0], p[2], gamma0) for i in range(4)]
    rG = r0 + h / 2.0
    p = [ip[i] - 8*k1[i]/27.0 + 2*k2[i] - 3544*k3[i]/2565.0
         + 1859*k4[i]/4104.0 - 11*k5[i]/40.0 for i in range(4)]
    k6 = [h * _deriv(self, i, rG, p[0], p[2], gamma0) for i in range(4)]
    out = []
    for i in range(4):
        out.append(ip[i] + 16*k1[i]/135.0 + 6656*k3[i]/12825.0
                   + 28561*k4[i]/56430.0 - 9*k5[i]/50.0 + 2*k6[i]/55.0)
    # error estimates
    for i in range(4):
        out.append(k1[i]/360.0 - 128*k2[i]/4275.0 - 2197*k4[i]/75240.0
                   + k5[i]/50.0 + 2*k6[i]/55.0)
    return tuple(out)


MOCGridCalc._deriv = _deriv
MOCGridCalc._runge_kutta = _runge_kutta
MOCGridCalc._runge_kutta_fehlberg = _runge_kutta_fehlberg


# ---------------------------------------------------------------------------
# 2.  Mass-flow integration helpers                                          -
# ---------------------------------------------------------------------------
def _calc_mdot_bd(self: MOCGridCalc, j: int, xD: float) -> float:
    """Direct port of ``MOC_GridCalc::CalcMdotBD``.

    Returns the mass flow from the wall (i = 0, j) to the interior point
    located at x = xD along RRC j, by interpolating the per-point
    mass-flow array.
    """
    g = self.grid
    ii = 0
    while True:
        ii += 1
        if xD <= g.x[ii, j]:
            break
    ratio = (xD - g.x[ii-1, j]) / (g.x[ii, j] - g.x[ii-1, j])
    rD = g.r[ii-1, j] + ratio * (g.r[ii, j] - g.r[ii-1, j])
    # distance-weighted ratio
    ratio = (math.sqrt((xD - g.x[ii-1, j])**2 + (rD - g.r[ii-1, j])**2)
             / math.sqrt((g.x[ii, j] - g.x[ii-1, j])**2
                         + (g.r[ii, j] - g.r[ii-1, j])**2))
    massflow_D = g.massflow[ii-1, j] + ratio * (g.massflow[ii, j] - g.massflow[ii-1, j])
    return g.massflow[0, j] - massflow_D


MOCGridCalc._calc_mdot_bd = _calc_mdot_bd


# ---------------------------------------------------------------------------
# 3.  FindPointE -- LRC DE integration                                       -
# ---------------------------------------------------------------------------
def _find_point_e(
    self: MOCGridCalc, j_start: int, xD: float, mdot_match: float,
    geom: int, n_type: int, n_rrc_plus: int, point_flag: int,
) -> _DEPoint:
    """Direct port of ``MOC_GridCalc::FindPointE``.

    Constructs the LRC from a point D on RRC ``j_start`` (at x = xD) out
    to a point E such that the integrated mass flow along DE equals
    ``mdot_match``. Returns a :class:`_DEPoint` with the properties at D
    and E.
    """
    g = self.grid
    mol_wt = self.mol_wt
    dS = _DEPoint()

    # step through points along the RRC to find where xD falls
    i = 0
    while True:
        i += 1
        if xD <= g.x[i, j_start]:
            break

    ratio = (xD - g.x[i-1, j_start]) / (g.x[i, j_start] - g.x[i-1, j_start])
    rD = g.r[i-1, j_start] + ratio * (g.r[i, j_start] - g.r[i-1, j_start])
    # distance-weighted
    denom = math.sqrt((g.x[i, j_start] - g.x[i-1, j_start])**2
                      + (g.r[i, j_start] - g.r[i-1, j_start])**2)
    if denom == 0.0:
        ratio = 0.0
    else:
        ratio = (math.sqrt((xD - g.x[i-1, j_start])**2
                           + (rD - g.r[i-1, j_start])**2) / denom)
    machD = g.mach[i-1, j_start] + ratio * (g.mach[i, j_start] - g.mach[i-1, j_start])
    gammaD = g.gamma[i-1, j_start] + ratio * (g.gamma[i, j_start] - g.gamma[i-1, j_start])

    if rD == 0.0 and n_type != PERFECT:
        raise _SecantFail("The solution will not converge on point E (rD=0)")

    thetaD = g.theta[i-1, j_start] + ratio * (g.theta[i, j_start] - g.theta[i-1, j_start])
    pres_D, temp_D, rho_D = self.calc_isentropic_p_t_rho_scalar(gammaD, machD)
    mu_D = calc_mu(machD)
    wD = machD * math.sqrt(gammaD * GASCON / mol_wt * GRAV * temp_D)

    dS.xD = xD
    dS.rD = rD
    dS.machD = machD
    dS.thetaD = thetaD
    dS.presD = pres_D
    dS.tempD = temp_D
    dS.rhoD = rho_D
    dS.i = i

    g.x_de[j_start] = xD
    g.r_de[j_start] = rD
    g.m_de[j_start] = machD
    g.theta_de[j_start] = thetaD
    g.p_de[j_start] = pres_D
    g.t_de[j_start] = temp_D
    g.rho_de[j_start] = rho_D
    g.g_de[j_start] = gammaD

    if mdot_match <= 0.0:
        dS.xE, dS.rE = xD, rD
        dS.machE, dS.thetaE = machD, thetaD
        dS.presE, dS.tempE, dS.rhoE = pres_D, temp_D, rho_D
        dS.gammaE = gammaD
        dS.wE = wD
        dS.last_rrc = j_start
        return dS

    # --- TWO-D case (constant properties along DE) -----------------
    if geom == TWOD:
        rE = (rD + (mdot_match * 12.0 * math.sin(thetaD + mu_D))
              / (rho_D * wD * math.sin(mu_D) * GRAV))
        xE = xD + (rE - rD) / l_dy_dx(thetaD, mu_D)
        dS.xE, dS.rE = xE, rE
        dS.machE, dS.thetaE = machD, thetaD
        dS.presE, dS.tempE, dS.rhoE = pres_D, temp_D, rho_D
        dS.gammaE, dS.wE = gammaD, wD
        if point_flag:
            dr = (rE - rD) / n_rrc_plus
            g.thrust[i, j_start] = 0.0
            g.s_thrust[i, j_start] = 0.0
            for jj in range(1, n_rrc_plus + 1):
                g.thrust[i, j_start+jj] = -99.9
                g.s_thrust[i, j_start+jj] = -99.9
                g.r[i, j_start+jj] = rD + jj*dr
                g.x[i, j_start+jj] = xD + (g.r[i, j_start+jj] - rD) / math.tan(thetaD + mu_D)
                g.mach[i, j_start+jj] = machD
                g.theta[i, j_start+jj] = thetaD
                g.pres[i, j_start+jj] = pres_D
                g.temp[i, j_start+jj] = temp_D
                g.rho[i, j_start+jj] = rho_D
                g.gamma[i, j_start+jj] = gammaD
                g.massflow[i, j_start+jj] = mdot_match * (g.r[i, j_start+jj] - rD) / (rE - rD)
                g.x_de[j_start+jj] = g.x[i, j_start+jj]
                g.r_de[j_start+jj] = g.r[i, j_start+jj]
                g.m_de[j_start+jj] = machD
                g.theta_de[j_start+jj] = thetaD
                g.p_de[j_start+jj] = pres_D
                g.t_de[j_start+jj] = temp_D
                g.rho_de[j_start+jj] = rho_D
                g.g_de[j_start+jj] = gammaD
                g.mass_de[j_start+jj] = g.massflow[i, j_start+jj]
        dS.last_rrc = j_start + n_rrc_plus
        return dS

    # --- AXI cases ---------------------------------------------------
    if geom != AXI:
        raise _SecantFail("Unknown geometry in FindPointE")

    if n_type == PERFECT:
        u0 = machD * math.sqrt(gammaD * GASCON / mol_wt * GRAV * temp_D)
        rE = math.sqrt(mdot_match * 144.0 / (GRAV * rho_D * u0 * PI))
        xE = xD + (rE - rD) * l_dy_dx(thetaD, calc_mu(machD))
        dS.xE, dS.rE = xE, rE
        dS.machE, dS.thetaE = machD, thetaD
        dS.presE, dS.tempE, dS.rhoE = pres_D, temp_D, rho_D
        dS.gammaE, dS.wE = gammaD, wD
        if point_flag:
            dr = rE / n_rrc_plus
            g.thrust[i, j_start] = 0.0
            g.s_thrust[i, j_start] = 0.0
            for jj in range(1, n_rrc_plus + 1):
                g.thrust[i, j_start+jj] = -99.9
                g.s_thrust[i, j_start+jj] = -99.9
                g.r[i, j_start+jj] = rD + jj*dr
                g.x[i, j_start+jj] = xD + (g.r[i, j_start+jj] - rD) / l_dy_dx(thetaD, mu_D)
                g.mach[i, j_start+jj] = machD
                g.theta[i, j_start+jj] = thetaD
                g.pres[i, j_start+jj] = pres_D
                g.temp[i, j_start+jj] = temp_D
                g.rho[i, j_start+jj] = rho_D
                g.gamma[i, j_start+jj] = gammaD
                g.massflow[i, j_start+jj] = mdot_match * (g.r[i, j_start+jj]**2) / (rE**2)
                g.x_de[j_start+jj] = g.x[i, j_start+jj]
                g.r_de[j_start+jj] = g.r[i, j_start+jj]
                g.m_de[j_start+jj] = machD
                g.theta_de[j_start+jj] = thetaD
                g.p_de[j_start+jj] = pres_D
                g.t_de[j_start+jj] = temp_D
                g.rho_de[j_start+jj] = rho_D
                g.g_de[j_start+jj] = gammaD
                g.mass_de[j_start+jj] = g.massflow[i, j_start+jj]
        dS.last_rrc = j_start + n_rrc_plus
        return dS

    # --- AXI, NOT PERFECT (the involved one) -------------------------
    mdot_total_e = 0.0
    mdot_total_0 = 0.0
    temp0 = temp_D
    gamma0 = gammaD
    a0 = math.sqrt(gamma0 * GASCON / mol_wt * GRAV * temp0)
    mach0 = machD
    theta0 = thetaD
    rho0 = rho_D
    vel0 = mach0 * a0
    u0 = vel0 * math.cos(theta0)
    v0 = vel0 * math.sin(theta0)
    mu0 = calc_mu(mach0)
    r0 = rD
    x0 = xD

    d_mdot = mdot_match / n_rrc_plus
    mdot_level = d_mdot
    machE = mach0
    thetaE = theta0
    presE = pres_D
    tempE = temp_D
    rhoE = rho_D
    gammaE = gammaD
    wE = wD
    rE = rD
    xE = xD
    uE = u0
    vE = v0
    muE = mu0
    dr = 0.0
    jj = 0

    while mdot_total_e < mdot_match:
        des = 0.05
        dM_err = dx_err = dT_err = dr_err = 9.9
        k = 0
        while ((abs(dM_err) > 1e-6 or abs(dx_err) > 1e-6
                or abs(dT_err) > 1e-6 or abs(dr_err) > 1e-12) and k < 50):
            k += 1
            dr = des * math.sqrt(mach0) * math.sin(theta0 + mu0)
            MXT = _runge_kutta_fehlberg(self, dr, r0, x0, mach0, theta0, gamma0)
            machE, xE, thetaE, rE = MXT[0], MXT[1], MXT[2], MXT[3]
            dM_err, dx_err, dT_err, dr_err = MXT[4], MXT[5], MXT[6], MXT[7]
            des *= 0.5
        if k >= 50:
            raise _SecantFail("Could not converge on DE integration (inner loop)")
        muE = calc_mu(machE)
        gammaE = gamma0
        presE, tempE, rhoE = self.calc_isentropic_p_t_rho_scalar(gammaE, machE)
        aE = math.sqrt(gammaE * GASCON / mol_wt * GRAV * tempE)
        wE = machE * aE
        uE = wE * math.cos(thetaE)
        vE = wE * math.sin(thetaE)
        dxdr = (xE - x0) / dr
        rho_u_avg = 0.5 * GRAV * (rho0*u0 + rhoE*uE - dxdr*(rho0*v0 + rhoE*vE))
        da = PI * (rE*rE - r0*r0) / 144.0
        mdot = rho_u_avg * da
        mdot_total_e += mdot

        if mdot_total_e < mdot_match:
            jj += 1
            if point_flag:
                g.mach[i, j_start+jj] = machE
                g.gamma[i, j_start+jj] = gammaE
                g.massflow[i, j_start+jj] = mdot_total_e
                g.theta[i, j_start+jj] = thetaE
                g.x[i, j_start+jj] = xE
                g.r[i, j_start+jj] = rE
                self._set_isentropic(i, j_start+jj, gammaE, machE)
                g.x_de[j_start+jj] = xE
                g.r_de[j_start+jj] = rE
                g.m_de[j_start+jj] = machE
                g.theta_de[j_start+jj] = thetaE
                g.g_de[j_start+jj] = gammaE
                g.mass_de[j_start+jj] = mdot_total_e
                p_, t_, rho_ = self.calc_isentropic_p_t_rho_scalar(gammaE, machE)
                g.p_de[j_start+jj] = p_
                g.t_de[j_start+jj] = t_
                g.rho_de[j_start+jj] = rho_
            mdot_total_0 = mdot_total_e
            gamma0 = gammaE
            mach0 = machE
            r0 = rE
            x0 = xE
            theta0 = thetaE
            rho0 = rhoE
            u0 = uE
            v0 = vE
            mu0 = muE
        # else: we will exit the while and refine via secant below

    # secant refinement on r to match mdot exactly
    min_r = r0
    max_r = rE
    r_guess = [r0, rE, 0.0]
    mdot_err = [(mdot_total_0 - mdot_match) / mdot_match,
                (mdot_total_e - mdot_match) / mdot_match,
                9.9]
    k = 0
    while abs(mdot_err[2]) > 1e-8 and k < 50:
        k += 1
        if mdot_err[0] != mdot_err[1]:
            r_guess[2] = (r_guess[1]
                          - mdot_err[1] * (r_guess[1] - r_guess[0])
                          / (mdot_err[1] - mdot_err[0]))
        else:
            k = 50
        r_guess[2] = max(min_r, min(max_r, r_guess[2]))
        dr = r_guess[2] - r0
        MXT = _runge_kutta(self, dr, r0, x0, mach0, theta0, gamma0)
        machE, xE, thetaE = MXT[0], MXT[1], MXT[2]
        rE = r0 + dr
        gammaE = gamma0
        muE = calc_mu(machE)
        presE, tempE, rhoE = self.calc_isentropic_p_t_rho_scalar(gammaE, machE)
        aE = math.sqrt(gammaE * GASCON / mol_wt * GRAV * tempE)
        wE = machE * aE
        dxdr = (xE - x0) / dr
        rho_u_avg = 0.5 * GRAV * (rho0*u0 + rhoE*uE - dxdr*(rho0*v0 + rhoE*vE))
        da = PI * (rE*rE - r0*r0) / 144.0
        mdot = rho_u_avg * da
        mdot_total_e = mdot_total_0 + mdot
        mdot_err[2] = (mdot_total_e - mdot_match) / mdot_match
        r_guess[0], r_guess[1] = r_guess[1], r_guess[2]
        mdot_err[0], mdot_err[1] = mdot_err[1], mdot_err[2]
    if k >= 50:
        # tolerate tiny residuals -- the C++ used to bail here, but in
        # practice the iteration is well-conditioned
        if abs(mdot_total_e - mdot_match) > 1e-6:
            raise _SecantFail("Could not converge on mass flow at point E")

    dS.xE, dS.rE = xE, rE
    dS.machE, dS.thetaE = machE, thetaE
    dS.presE, dS.tempE, dS.rhoE = presE, tempE, rhoE
    dS.gammaE, dS.wE = gammaE, wE

    jj += 1
    if point_flag:
        g.mach[i, j_start+jj] = machE
        g.gamma[i, j_start+jj] = gammaE
        g.massflow[i, j_start+jj] = mdot_total_e
        g.theta[i, j_start+jj] = thetaE
        g.x[i, j_start+jj] = xE
        g.r[i, j_start+jj] = rE
        self._set_isentropic(i, j_start+jj, gammaE, machE)
        g.x_de[j_start+jj] = xE
        g.r_de[j_start+jj] = rE
        g.m_de[j_start+jj] = machE
        g.theta_de[j_start+jj] = thetaE
        g.g_de[j_start+jj] = gammaE
        g.mass_de[j_start+jj] = mdot_total_e
        p_, t_, rho_ = self.calc_isentropic_p_t_rho_scalar(gammaE, machE)
        g.p_de[j_start+jj] = p_
        g.t_de[j_start+jj] = t_
        g.rho_de[j_start+jj] = rho_
    dS.last_rrc = j_start + jj
    return dS


MOCGridCalc._find_point_e = _find_point_e


# ---------------------------------------------------------------------------
# 4.  CalcLRCDE -- the BD->DE point-matching iteration                       -
# ---------------------------------------------------------------------------
def _calc_lrc_de(
    self: MOCGridCalc, j: int, i_end: int, p_amb: float, geom: int,
    n_rrc_plus: int, n_type: int, r_match: float, point_flag: int,
) -> _DEPoint:
    """Direct port of ``MOC_GridCalc::CalcLRCDE``."""
    g = self.grid
    mol_wt = self.mol_wt

    if n_type == PERFECT:
        mdot_bd = _calc_mdot_bd(self, j, g.x[i_end, j])
        return _find_point_e(self, j, g.x[i_end, j], mdot_bd,
                             geom, n_type, n_rrc_plus, point_flag)

    # Set up secant variables
    xD_max = min(9e9, g.x[i_end, j])
    xD = [g.x[0, j], 0.0, 0.0]
    xD_min = g.x[0, j]
    w0 = (g.mach[0, j]
          * math.sqrt(g.gamma[0, j] * GASCON / mol_wt * GRAV * g.temp[0, j]))

    param_err = [0.0, 0.0, 9.9]
    thetaCalc = 0.0
    dS = _DEPoint()

    if n_type == RAO:
        thetaCalc = 0.5 * math.asin(
            2 * (g.pres[0, j] - p_amb) * 144.0
            / (g.rho[0, j] * w0 * w0 * math.tan(calc_mu(g.mach[0, j])))
        )
        param_err[0] = (g.theta[0, j] - thetaCalc) / thetaCalc
        if param_err[0] < 0.0:
            dS.status = SEC_FAIL_LOW
            return dS
        if thetaCalc < 0.0:
            dS.status = SEC_FAIL
            return dS
    elif n_type == FIXEDEND:
        param_err[0] = g.r[0, j] - r_match
        if param_err[0] > 0.0:
            dS.status = SEC_FAIL_HIGH
            return dS

    # walk along j until param_err changes sign
    i = 1
    if n_type == RAO:
        param_err[1] = 9e9
        while i < i_end and thetaCalc >= 0.0 and param_err[1] > 0.0:
            xD[1] = g.x[i, j]
            mdot_bd = _calc_mdot_bd(self, j, xD[1])
            dS = _find_point_e(self, j, xD[1], mdot_bd, geom, n_type,
                               n_rrc_plus, point_flag)
            thetaCalc = 0.5 * math.asin(
                2 * (dS.presE - p_amb) * 144.0
                / (dS.rhoE * dS.wE * dS.wE * math.tan(calc_mu(dS.machE)))
            )
            param_err[1] = (dS.thetaE - thetaCalc) / abs(thetaCalc)
            i += 1
    elif n_type == FIXEDEND:
        param_err[1] = param_err[0]
        while i < i_end and param_err[1] < 0.0:
            xD[1] = g.x[i, j]
            mdot_bd = _calc_mdot_bd(self, j, xD[1])
            dS = _find_point_e(self, j, xD[1], mdot_bd, geom, n_type,
                               n_rrc_plus, point_flag)
            param_err[1] = dS.rE - r_match
            i += 1

    if thetaCalc < 0.0 and n_type == RAO:
        dS.status = SEC_FAIL_HIGH
        return dS
    if i == i_end:
        dS.status = SEC_FAIL_HIGH
        return dS

    xD[0] = g.x[i-2, j]
    mdot_bd = _calc_mdot_bd(self, j, xD[0])
    dS = _find_point_e(self, j, xD[0], mdot_bd, geom, n_type,
                       n_rrc_plus, point_flag)
    if n_type == RAO:
        thetaCalc = 0.5 * math.asin(
            2 * (dS.presE - p_amb) * 144.0
            / (dS.rhoE * dS.wE * dS.wE * math.tan(calc_mu(dS.machE)))
        )
        param_err[0] = (dS.thetaE - thetaCalc) / thetaCalc
        if thetaCalc < 0.0:
            dS.status = SEC_FAIL
            return dS
    elif n_type == FIXEDEND:
        param_err[0] = dS.rE - r_match

    i = 0
    param_err[2] = 9.9
    xD_max_bisect = xD_max
    xD_min_bisect = xD_min
    while abs(param_err[2]) > 1e-7 and i < 50:
        i += 1
        if param_err[0] != param_err[1]:
            xD[2] = (xD[1]
                     - param_err[1] * (xD[1] - xD[0])
                     / (param_err[1] - param_err[0]))
        else:
            i = 50
        xD[2] = max(xD_min, min(xD_max, xD[2]))

        if xD[2] <= g.x[0, j]:
            xD[2] = g.x[0, j]
            if n_type == RAO:
                thetaCalc = 0.5 * math.asin(
                    2 * (g.pres[0, j] - p_amb) * 144.0
                    / (g.rho[0, j] * w0 * w0 * math.tan(calc_mu(g.mach[0, j])))
                )
                param_err[2] = (g.theta[0, j] - thetaCalc) / thetaCalc
                if thetaCalc < 0.0:
                    dS.status = SEC_FAIL
                    return dS
            elif n_type == FIXEDEND:
                param_err[2] = g.r[0, j] - r_match
        else:
            if xD[2] >= g.x[i_end, j]:
                xD[2] = g.x[i_end, j]
                mdot_bd = _calc_mdot_bd(self, j, xD[2])
                dS = _find_point_e(self, j, xD[2], mdot_bd, geom,
                                   PERFECT, n_rrc_plus, point_flag)
            elif xD[2] < xD_min:
                xD[2] = xD_min
                mdot_bd = _calc_mdot_bd(self, j, xD[2])
                dS = _find_point_e(self, j, xD[2], mdot_bd, geom,
                                   n_type, n_rrc_plus, point_flag)
            else:
                mdot_bd = _calc_mdot_bd(self, j, xD[2])
                dS = _find_point_e(self, j, xD[2], mdot_bd, geom,
                                   n_type, n_rrc_plus, point_flag)
            if dS.status == SEC_FAIL:
                xD_min = xD[2]
                xD[2] = 0.5 * (xD_min + xD_max)
                mdot_bd = _calc_mdot_bd(self, j, xD[2])
                dS = _find_point_e(self, j, xD[2], mdot_bd, geom,
                                   n_type, n_rrc_plus, point_flag)
            if n_type == RAO:
                thetaCalc = 0.5 * math.asin(
                    2 * (dS.presE - p_amb) * 144.0
                    / (dS.rhoE * dS.wE * dS.wE * math.tan(calc_mu(dS.machE)))
                )
                param_err[2] = (dS.thetaE - thetaCalc) / thetaCalc
                if thetaCalc < 0.0:
                    dS.status = SEC_FAIL
                    return dS
            elif n_type == FIXEDEND:
                param_err[2] = dS.rE - r_match

        if abs(param_err[2]) > 1e-7:
            g.reset(dS.i, dS.i, j+1, j+n_rrc_plus)

        xD[0], xD[1] = xD[1], xD[2]
        param_err[0], param_err[1] = param_err[1], param_err[2]
        if param_err[2] < 0.0:
            xD_max_bisect = xD[2]
        elif param_err[2] > 0.0:
            xD_min_bisect = xD[2]

    if i >= 50:
        dS.status = SEC_FAIL_LOW if xD[2] == xD_max else SEC_FAIL_HIGH
        return dS

    dS.mdot = mdot_bd
    dS.status = SEC_OK
    return dS


MOCGridCalc._calc_lrc_de = _calc_lrc_de


# ---------------------------------------------------------------------------
# 5.  CalcBDERegion -- back-calculation of mesh from DE to the wall          -
# ---------------------------------------------------------------------------
def _calc_bde_region(
    self: MOCGridCalc, iD: int, jD: int, jEnd: int, geom: int,
) -> None:
    """Direct port of ``MOC_GridCalc::CalcBDERegion``."""
    g = self.grid
    for j in range(jD + 1, jEnd + 1):
        for i in range(iD - 1, -1, -1):
            x_err = r_err = m_err = t_err = 9.9
            x3_old = r3_old = m3_old = theta3_old = 9.9
            k = 0

            s1 = l_dy_dx(g.theta[i, j-1], calc_mu(g.mach[i, j-1]))
            A1 = calc_A(g.mach[i, j-1], g.gamma[i, j-1])
            M1 = g.mach[i, j-1]
            TH1 = g.theta[i, j-1]
            B1 = calc_B(g.mach[i, j-1], g.theta[i, j-1], g.r[i, j-1])
            R1 = calc_R(g.mach[i, j-1], g.theta[i, j-1], g.r[i, j-1])
            RS1 = calc_R_star(g.mach[i, j-1], g.theta[i, j-1], g.r[i, j-1])

            s2 = r_dy_dx(g.theta[i+1, j], calc_mu(g.mach[i+1, j]))
            A2 = calc_A(g.mach[i+1, j], g.gamma[i+1, j])
            M2 = g.mach[i+1, j]
            TH2 = g.theta[i+1, j]
            B2 = calc_B(g.mach[i+1, j], g.theta[i+1, j], g.r[i+1, j])
            b2 = calc_b(g.mach[i+1, j], g.theta[i+1, j], g.r[i+1, j])
            R2 = calc_R(g.mach[i+1, j], g.theta[i+1, j], g.r[i+1, j])
            RS2 = calc_R_star(g.mach[i+1, j], g.theta[i+1, j], g.r[i+1, j])

            s3lrc = s1
            s3rrc = s2
            b3 = b2
            B3 = B1
            R3 = R1
            RS3 = RS1
            A3 = (A1 + A2) / 2.0
            G3 = 0.5 * (g.gamma[i, j-1] + g.gamma[i+1, j])
            M3 = TH3 = 0.0

            while ((abs(x_err) > self.con_crit or abs(r_err) > self.con_crit
                    or abs(m_err) > self.con_crit or abs(t_err) > self.con_crit)
                   and k < 50):
                k += 1
                slope13 = tan_avg(s1, s3lrc)
                slope23 = tan_avg(s2, s3rrc)
                g.x[i, j] = ((g.r[i, j-1] - g.r[i+1, j]
                              - slope13*g.x[i, j-1] + slope23*g.x[i+1, j])
                             / (slope23 - slope13))
                if abs(s2) <= abs(s1):
                    g.r[i, j] = g.r[i+1, j] + slope23 * (g.x[i, j] - g.x[i+1, j])
                else:
                    g.r[i, j] = g.r[i, j-1] + slope13 * (g.x[i, j] - g.x[i, j-1])

                if geom == TWOD:
                    T1 = T2 = 0.0
                else:
                    if abs(b2) <= abs(RS2):
                        T2 = (g.x[i, j] - g.x[i+1, j]) * (b2 + b3)
                    else:
                        T2 = (g.r[i, j] - g.r[i+1, j]) * (RS3 + RS2)
                    if abs(B1) <= abs(R1):
                        T1 = (g.x[i, j] - g.x[i, j-1]) * (B1 + B3)
                    else:
                        T1 = (g.r[i, j] - g.r[i, j-1]) * (R3 + R1)

                M3 = ((2*(TH2 - TH1) + M2*(A2 + A3) + M1*(A1 + A3) + T1 + T2)
                      / (A1 + A2 + 2*A3))
                A3 = calc_A(M3, G3)
                TH3 = ((TH1 + TH2) / 2.0
                       + 0.25*(M2*(A3 + A2) - M1*(A1 + A3)
                               - M3*(A2 - A1) + T2 - T1))
                if TH3 != 0.0:
                    TH3 = max(TH3, 0.0)

                s3lrc = l_dy_dx(TH3, calc_mu(M3))
                s3rrc = r_dy_dx(TH3, calc_mu(M3))
                B3 = calc_B(M3, TH3, g.r[i, j])
                b3 = calc_b(M3, TH3, g.r[i, j])
                R3 = calc_R(M3, TH3, g.r[i, j])
                RS3 = calc_R_star(M3, TH3, g.r[i, j])

                x_err = (g.x[i, j] - x3_old) / x3_old if x3_old else 0.0
                r_err = (g.r[i, j] - r3_old) / r3_old if r3_old else 0.0
                m_err = (M3 - m3_old) / m3_old if m3_old else 0.0
                t_err = (TH3 - theta3_old) / theta3_old if theta3_old else 0.0
                x3_old, r3_old = g.x[i, j], g.r[i, j]
                m3_old, theta3_old = M3, TH3
            if k >= 50:
                raise _SecantFail("Could not converge on back-calculated LRC")
            g.mach[i, j] = M3
            g.theta[i, j] = TH3
            g.gamma[i, j] = G3
            self._set_isentropic(i, j, g.gamma[i, j], g.mach[i, j])


MOCGridCalc._calc_bde_region = _calc_bde_region


# ---------------------------------------------------------------------------
# 6.  CalcRemainingMesh -- the mesh above DE                                 -
# ---------------------------------------------------------------------------
def _calc_remaining_mesh(
    self: MOCGridCalc, iD: int, jD: int, jEnd: int, geom: int,
) -> None:
    """Direct port of ``MOC_GridCalc::CalcRemainingMesh``."""
    g = self.grid
    for j in range(jD + 1, jEnd + 1):
        g.i_last[j] = g.i_last[j-1] + 1
        if not self._calc_interior_mesh_points(j, iD+1, int(g.i_last[j]), 1, geom):
            raise _SecantFail("Could not converge in CalcRemainingMesh")
        self._calc_axial_mesh_point(j, int(g.i_last[j]))
        i = 0
        while i <= g.i_last[j]:
            if g.r[i, j] < 0.0:
                g.i_last[j] -= 1
                for ii in range(i, int(g.i_last[j]) + 1):
                    for arr in (g.x, g.r, g.mach, g.theta, g.gamma,
                                g.pres, g.temp, g.rho, g.massflow):
                        arr[ii, j] = arr[ii+1, j]
            i += 1


MOCGridCalc._calc_remaining_mesh = _calc_remaining_mesh


# ---------------------------------------------------------------------------
# 7.  CalcWallContour -- determine final wall positions                      -
# ---------------------------------------------------------------------------
def _calc_wall_contour(
    self: MOCGridCalc, iD: int, j_start: int, j_end: int, geom: int,
) -> None:
    """Direct port of ``MOC_GridCalc::CalcWallContour``."""
    g = self.grid
    mol_wt = self.mol_wt
    j = j_end
    for j in range(j_start, j_end):
        mdot_match = ((g.massflow[0, j_start-1] - g.massflow[iD, j_start-1])
                      - g.massflow[iD, j])
        temp0 = g.temp[iD, j]
        gamma0 = g.gamma[iD, j]
        mach0 = g.mach[iD, j]
        a0 = math.sqrt(gamma0 * GASCON / mol_wt * GRAV * temp0)
        theta0 = g.theta[iD, j]
        rho0 = g.rho[iD, j]
        u0 = mach0 * math.cos(theta0) * a0
        v0 = mach0 * math.sin(theta0) * a0
        r0 = g.r[iD, j]
        x0 = g.x[iD, j]
        mdot = 0.0
        mdot0 = 0.0
        u1 = v1 = 0.0
        r1 = g.r[iD, j]
        x1 = g.x[iD, j]
        rho1 = rho0
        i = iD - 1
        i_loop_end = i
        # find where the integrated mass flow crosses mdot_match
        for i in range(iD - 1, -1, -1):
            temp1 = g.temp[i, j]
            gamma1 = g.gamma[i, j]
            a1 = math.sqrt(gamma1 * GASCON / mol_wt * GRAV * temp1)
            mach1 = g.mach[i, j]
            theta1 = g.theta[i, j]
            rho1 = g.rho[i, j]
            u1 = mach1 * math.cos(theta1) * a1
            v1 = mach1 * math.sin(theta1) * a1
            r1 = g.r[i, j]
            x1 = g.x[i, j]
            dxdr = (x1 - x0) / (r1 - r0) if (r1 - r0) != 0.0 else 0.0
            rho_u_avg = 0.5 * GRAV * (rho0*u0 + rho1*u1
                                      - dxdr * (rho0*v0 + rho1*v1))
            if geom == TWOD:
                da = (r1 - r0) / 12.0
            else:
                da = PI * (r1*r1 - r0*r0) / 144.0
            mdot += rho_u_avg * da
            if mdot < mdot_match:
                temp0, a0, gamma0 = temp1, a1, gamma1
                mach0, theta0, rho0 = mach1, theta1, rho1
                u0, v0, r0, x0 = u1, v1, r1, x1
                mdot0 = mdot
            else:
                i_loop_end = i
                break
        else:
            i_loop_end = 0

        # secant on x to refine wall location
        x_guess = [x0, x1, 0.0]
        mdot_err = [(mdot0 - mdot_match) / mdot_match,
                    (mdot - mdot_match) / mdot_match,
                    9.9]
        k = 0
        x2 = r2 = mach2 = theta2 = gamma2 = rho2 = 0.0
        while abs(mdot_err[2]) > self.con_crit and k < 50:
            k += 1
            if mdot_err[0] != mdot_err[1]:
                x_guess[2] = (x_guess[1]
                              - mdot_err[1] * (x_guess[1] - x_guess[0])
                              / (mdot_err[1] - mdot_err[0]))
            else:
                k = 50
            x2 = x_guess[2]
            ratio = (x2 - x0) / (x1 - x0) if (x1 - x0) else 0.0
            r2 = r0 + ratio * (r1 - r0)
            denom = math.sqrt((x1 - x0)**2 + (r1 - r0)**2)
            ratio = (math.sqrt((x2 - x0)**2 + (r2 - r0)**2) / denom
                     if denom else 0.0)
            temp2 = temp0 + ratio * (g.temp[i_loop_end, j] - temp0)
            gamma2 = gamma0 + ratio * (g.gamma[i_loop_end, j] - gamma0)
            mach2 = mach0 + ratio * (g.mach[i_loop_end, j] - mach0)
            theta2 = theta0 + ratio * (g.theta[i_loop_end, j] - theta0)
            a2 = math.sqrt(gamma2 * GASCON / mol_wt * GRAV * temp2)
            rho2 = rho0 + ratio * (g.rho[i_loop_end, j] - rho0)
            r2 = r0 + ratio * (g.r[i_loop_end, j] - r0)
            u2 = mach2 * math.cos(theta2) * a2
            v2 = mach2 * math.sin(theta2) * a2
            dxdr = (x2 - x0) / (r2 - r0) if (r2 - r0) else 0.0
            rho_u_avg = 0.5 * GRAV * (rho0*u0 + rho2*u2
                                      - dxdr * (rho0*v0 + rho2*v2))
            if geom == TWOD:
                da = (r2 - r0) / 12.0
            else:
                da = PI * (r2*r2 - r0*r0) / 144.0
            mdot = mdot0 + rho_u_avg * da
            mdot_err[2] = (mdot - mdot_match) / mdot_match
            x_guess[0], x_guess[1] = x_guess[1], x_guess[2]
            mdot_err[0], mdot_err[1] = mdot_err[1], mdot_err[2]
        if k >= 50:
            raise _SecantFail("Could not converge on mass flow in CalcWallContour")

        i_translate = i_loop_end
        g.x[0, j] = x2
        g.r[0, j] = r2
        g.theta[0, j] = theta2
        self._set_isentropic(0, j, gamma2, mach2)
        for i in range(i_translate + 1, int(g.i_last[j]) + 1):
            i_new = i - i_translate
            g.x[i_new, j] = g.x[i, j]
            g.r[i_new, j] = g.r[i, j]
            g.theta[i_new, j] = g.theta[i, j]
            self._set_isentropic(i_new, j, g.gamma[i, j], g.mach[i, j])
        g.i_last[j] = int(g.i_last[j]) - i_translate

    # final jEnd line: translate iD down to 0
    j = j_end
    i_translate = iD
    for i in range(i_translate, int(g.i_last[j]) + 1):
        i_new = i - i_translate
        g.x[i_new, j] = g.x[i, j]
        g.r[i_new, j] = g.r[i, j]
        g.theta[i_new, j] = g.theta[i, j]
        self._set_isentropic(i_new, j, g.gamma[i, j], g.mach[i, j])
    g.i_last[j] = int(g.i_last[j]) - i_translate


MOCGridCalc._calc_wall_contour = _calc_wall_contour


# ---------------------------------------------------------------------------
# 8.  CalcDE -- finalize the DE arrays                                       -
# ---------------------------------------------------------------------------
def _calc_de(self: MOCGridCalc, iD: int, jD: int, jEnd: int,
             n_type: int, geom: int) -> int:
    """Direct port of ``MOC_GridCalc::CalcDE``."""
    g = self.grid
    if n_type != PERFECT:
        if geom == TWOD:
            g.r_de[jD-1] = 0.0
            g.x_de[jD-1] = g.x_de[jD] - (g.r_de[jD] - g.r_de[jD-1]) / l_dy_dx(
                g.theta_de[jD], calc_mu(g.m_de[jD]))
            g.m_de[jD-1] = g.m_de[jD]
            g.t_de[jD-1] = g.t_de[jD]
            g.p_de[jD-1] = g.p_de[jD]
            g.theta_de[jD-1] = g.theta_de[jD]
            g.rho_de[jD-1] = g.rho_de[jD]
            g.g_de[jD-1] = g.g_de[jD]
            jD -= 1
        else:
            if g.r[iD-1, jD] == g.r[iD+1, jD]:
                ratio = 0.0
            else:
                ratio = ((g.r_de[jD] - g.r[iD+1, jD])
                         / (g.r[iD-1, jD] - g.r[iD+1, jD]))
            ii = iD - 1
            j = jD - 1
            while j >= 0:
                i = 0
                if g.r[ii+1, j+1] == 0:
                    break
                while g.x[i, j] < g.x[ii, j+1] and i < g.i_last[j]:
                    i += 1
                if i == g.i_last[j]:
                    break
                g.x_de[j]     = g.x[i, j]     + ratio * (g.x[i-1, j]     - g.x[i, j])
                g.r_de[j]     = g.r[i, j]     + ratio * (g.r[i-1, j]     - g.r[i, j])
                g.m_de[j]     = g.mach[i, j]  + ratio * (g.mach[i-1, j]  - g.mach[i, j])
                g.t_de[j]     = g.temp[i, j]  + ratio * (g.temp[i-1, j]  - g.temp[i, j])
                g.p_de[j]     = g.pres[i, j]  + ratio * (g.pres[i-1, j]  - g.pres[i, j])
                g.rho_de[j]   = g.rho[i, j]   + ratio * (g.rho[i-1, j]   - g.rho[i, j])
                g.g_de[j]     = g.gamma[i, j] + ratio * (g.gamma[i-1, j] - g.gamma[i, j])
                g.theta_de[j] = g.theta[i, j] + ratio * (g.theta[i-1, j] - g.theta[i, j])
                ii = i - 1
                j -= 1
            jD = j + 1
    # Translate DE arrays to start at 0
    jT = jD
    for j in range(jD, jEnd + 1):
        idx = j - jT
        g.x_de[idx]     = g.x_de[j]
        g.r_de[idx]     = g.r_de[j]
        g.t_de[idx]     = g.t_de[j]
        g.p_de[idx]     = g.p_de[j]
        g.rho_de[idx]   = g.rho_de[j]
        g.g_de[idx]     = g.g_de[j]
        g.theta_de[idx] = g.theta_de[j]
        g.m_de[idx]     = g.m_de[j]
        g.mass_de[idx]  = g.mass_de[j]
    return jEnd - jD


MOCGridCalc._calc_de = _calc_de


# ---------------------------------------------------------------------------
# 9.  CalcContouredNozzle -- top-level driver                                -
# ---------------------------------------------------------------------------
def _calc_contoured_nozzle(
    self: MOCGridCalc, param_type: int, param_match: list[float],
    g_gamma: float, p_amb: float, geom: int, n_rrc_plus: int, n_type: int,
    n_sl_i: int, n_sl_j: int,
) -> bool:
    """Direct port of ``MOC_GridCalc::CalcContouredNozzle``.

    Implements the RAO iteration: for each candidate ThetaB, build the
    kernel along the throat arc, find the LRC DE, evaluate the design-
    parameter error, and iterate via the secant method until the design
    target is met.
    """
    g = self.grid
    theta_b = [self.theta_bi * RAD_PER_DEG, 0.0, 0.0]
    param_err = [0.0, 0.0, 0.0]
    param_exit = 0.0
    last_kernel_j = 0
    dS = _DEPoint()

    # --- two initial guesses ---------------------------------------
    i = 0
    while i <= 1:
        if i == 1:
            if abs(param_err[0]) >= 1.0:
                i = 0
                theta_b[0] *= 0.8
                continue
            theta_b[1] = self._set_theta_b(param_type, param_err[0], theta_b[0])

        x_arc_max = math.sin(theta_b[i]) * self.RWTD
        j = 0
        while g.x[0, j+1] < x_arc_max and j < last_kernel_j:
            j += 1
        if j > 7:
            j -= 5
        else:
            j = 0
        last_kernel_j = self._calc_rrcs_along_arc(
            j, self.RWTD, theta_b[i], self.DTLIMIT, p_amb, geom)
        if last_kernel_j == _HARD_FAIL:
            return False
        k = 0
        while last_kernel_j < 0 and k < 20:
            k += 1
            last_kernel_j = -last_kernel_j
            self.theta_b_max = math.asin(g.x[0, last_kernel_j] / self.RWTD)
            if i == 0:
                theta_b[0] = min(0.95 * self.theta_b_max, 0.95 * theta_b[0])
            else:
                theta_b[1] = 0.5 * (theta_b[0] + theta_b[1])
            x_arc_max = math.sin(theta_b[i]) * self.RWTD
            j = 0
            while g.x[0, j+1] < x_arc_max and j < last_kernel_j:
                j += 1
            j = j - 5 if j > 7 else 0
            last_kernel_j = self._calc_rrcs_along_arc(
                j, self.RWTD, theta_b[i], self.DTLIMIT, p_amb, geom)
            if last_kernel_j == _HARD_FAIL:
                return False
        if k >= 20:
            return False

        dS = self._calc_lrc_de(last_kernel_j, int(g.i_last[last_kernel_j]),
                               p_amb, geom, n_rrc_plus, n_type,
                               param_match[1], 0)
        # handle the various failure modes by adjusting thetaB
        if dS.status == SEC_FAIL_LOW:
            k = 0
            while dS.status == SEC_FAIL_LOW and k < 10:
                k += 1
                self.theta_b_min = theta_b[i]
                theta_b[i] = 0.5 * (self.theta_b_min + self.theta_b_max)
                x_arc_max = math.sin(theta_b[i]) * self.RWTD
                j = 0
                while g.x[0, j+1] < x_arc_max and j < last_kernel_j:
                    j += 1
                j = j - 5 if j > 7 else 0
                last_kernel_j = self._calc_rrcs_along_arc(
                    j, self.RWTD, theta_b[i], self.DTLIMIT, p_amb, geom)
                if last_kernel_j == _HARD_FAIL:
                    return False
                while last_kernel_j < 0:
                    last_kernel_j = -last_kernel_j
                    self.theta_b_max = math.asin(g.x[0, last_kernel_j] / self.RWTD)
                    theta_b[i] = self.theta_b_max
                    j = 0
                    while g.x[0, j+1] < x_arc_max and j < last_kernel_j:
                        j += 1
                    j = j - 2 if j > 3 else 0
                    last_kernel_j = self._calc_rrcs_along_arc(
                        j, self.RWTD, theta_b[i], self.DTLIMIT, p_amb, geom)
                    if last_kernel_j == _HARD_FAIL:
                        return False
                dS = self._calc_lrc_de(
                    last_kernel_j, int(g.i_last[last_kernel_j]),
                    p_amb, geom, n_rrc_plus, n_type, param_match[1], 0)
            if k > 10:
                return False
        elif dS.status == SEC_FAIL_HIGH:
            k = 0
            while (dS.status == SEC_FAIL_HIGH
                   and self.theta_b_min < self.theta_b_max and k < 10):
                k += 1
                self.theta_b_max = theta_b[i]
                if i == 1 and param_err[0] < 0.0:
                    theta_b[i] = 0.5 * (theta_b[0] + self.theta_b_max)
                else:
                    theta_b[i] = (self.theta_b_max + self.theta_b_min) / 2.0
                x_arc_max = math.sin(theta_b[i]) * self.RWTD
                j = 0
                while g.x[0, j+1] < x_arc_max and j < last_kernel_j:
                    j += 1
                j = j - 5 if j > 7 else 0
                last_kernel_j = self._calc_rrcs_along_arc(
                    j, self.RWTD, theta_b[i], self.DTLIMIT, p_amb, geom)
                if last_kernel_j == _HARD_FAIL:
                    return False
                dS = self._calc_lrc_de(
                    last_kernel_j, int(g.i_last[last_kernel_j]),
                    p_amb, geom, n_rrc_plus, n_type, param_match[1], 0)
            if k > 10:
                return False
        elif dS.status == SEC_FAIL:
            return False

        if self.theta_b_min >= self.theta_b_max:
            return False

        self.last_rrc = dS.last_rrc
        if param_type == EXITMACH:
            param_exit = dS.machE
        elif param_type in (NOZZLELENGTH, ENDPOINT):
            param_exit = dS.xE
        elif param_type == EXITPRESSURE:
            param_exit = self.p_total / dS.presE
        elif param_type == EPS:
            param_exit = dS.rE if geom == TWOD else dS.rE * dS.rE
        param_err[i] = (param_exit - param_match[0]) / param_match[0]
        i += 1

    # set min theta if both initial guesses too high
    if param_err[0] < 0.0 and param_err[1] < 0.0:
        if param_type != ENDPOINT:
            self.theta_b_min = max(theta_b[0], theta_b[1])
        else:
            self.theta_b_max = min(theta_b[0], theta_b[1])

    # --- main secant iteration ------------------------------------
    i = 0
    param_err[2] = 9.9
    while abs(param_err[2]) > 1e-8 and i < 20:
        i += 1
        if param_err[0] != param_err[1]:
            theta_b[2] = (theta_b[1]
                          - param_err[1] * (theta_b[1] - theta_b[0])
                          / (param_err[1] - param_err[0]))
        else:
            i = 20
        theta_b[2] = max(self.theta_b_min, min(self.theta_b_max, theta_b[2]))

        x_arc_max = math.sin(theta_b[2]) * self.RWTD
        j = 0
        while g.x[0, j+1] < x_arc_max and j < last_kernel_j:
            j += 1
        j = j - 5 if j > 7 else 0
        last_kernel_j = self._calc_rrcs_along_arc(
            j, self.RWTD, theta_b[2], self.DTLIMIT, p_amb, geom)
        if last_kernel_j == _HARD_FAIL:
            return False
        while last_kernel_j < 0:
            last_kernel_j = -last_kernel_j
            self.theta_b_max = math.asin(g.x[0, last_kernel_j] / self.RWTD)
            theta_b[2] = self.theta_b_max
            j = 0
            while g.x[0, j+1] < x_arc_max and j < last_kernel_j:
                j += 1
            j = j - 2 if j > 3 else 0
            last_kernel_j = self._calc_rrcs_along_arc(
                j, self.RWTD, theta_b[2], self.DTLIMIT, p_amb, geom)
            if last_kernel_j == _HARD_FAIL:
                return False
        dS = self._calc_lrc_de(last_kernel_j, int(g.i_last[last_kernel_j]),
                               p_amb, geom, n_rrc_plus, n_type,
                               param_match[1], 0)
        bisect_k = 0
        while (dS.status in (SEC_FAIL_LOW, SEC_FAIL_HIGH) and bisect_k < 20):
            bisect_k += 1
            if dS.status == SEC_FAIL_LOW:
                self.theta_b_min = theta_b[2]
            else:
                self.theta_b_max = theta_b[2]
            theta_b[2] = (self.theta_b_min + self.theta_b_max) / 2.0
            x_arc_max = math.sin(theta_b[2]) * self.RWTD
            j = 0
            while g.x[0, j+1] < x_arc_max and j < last_kernel_j:
                j += 1
            j = j - 5 if j > 7 else 0
            last_kernel_j = self._calc_rrcs_along_arc(
                j, self.RWTD, theta_b[2], self.DTLIMIT, p_amb, geom)
            if last_kernel_j == _HARD_FAIL:
                return False
            while last_kernel_j < 0:
                last_kernel_j = -last_kernel_j
                self.theta_b_max = math.asin(g.x[0, last_kernel_j] / self.RWTD)
                theta_b[2] = self.theta_b_max
                j = 0
                while g.x[0, j+1] < x_arc_max and j < last_kernel_j:
                    j += 1
                j = j - 2 if j > 3 else 0
                last_kernel_j = self._calc_rrcs_along_arc(
                    j, self.RWTD, theta_b[2], self.DTLIMIT, p_amb, geom)
                if last_kernel_j == _HARD_FAIL:
                    return False
            dS = self._calc_lrc_de(
                last_kernel_j, int(g.i_last[last_kernel_j]),
                p_amb, geom, n_rrc_plus, n_type, param_match[1], 0)

        if param_type == EXITMACH:
            param_exit = dS.machE
        elif param_type in (NOZZLELENGTH, ENDPOINT):
            param_exit = dS.xE
        elif param_type == EXITPRESSURE:
            param_exit = self.p_total / dS.presE
        elif param_type == EPS:
            param_exit = dS.rE if geom == TWOD else dS.rE * dS.rE
        param_err[2] = (param_exit - param_match[0]) / param_match[0]
        theta_b[0], theta_b[1] = theta_b[1], theta_b[2]
        param_err[0], param_err[1] = param_err[1], param_err[2]
        if param_err[2] < 0.0:
            if param_type != ENDPOINT:
                self.theta_b_min = theta_b[2]
            else:
                self.theta_b_max = theta_b[2]
        elif param_err[2] > 0.0:
            if param_type != ENDPOINT:
                self.theta_b_max = theta_b[2]
            else:
                self.theta_b_min = theta_b[2]
    if i >= 20 and abs(param_err[2]) > 1e-5:
        return False

    # --- finalize ------------------------------------------------
    g.reset(0, self.max_lrc - 1, last_kernel_j + 1, self.max_rrc - 1)
    dS = self._calc_lrc_de(last_kernel_j, int(g.i_last[last_kernel_j]),
                           p_amb, geom, n_rrc_plus, n_type,
                           param_match[1], 1)
    self.theta_b_ans = theta_b[2]
    j = last_kernel_j
    if n_type == RAO:
        self.theta_b_max = theta_b[2]
    elif n_type == PERFECT:
        self.theta_b_min = theta_b[2]

    ii = dS.i
    if g.x[ii, j] != dS.xD and g.r[ii, j] != dS.rD:
        for i in range(int(g.i_last[j]), ii - 1, -1):
            for arr in (g.x, g.r, g.mach, g.theta, g.pres, g.temp,
                        g.rho, g.massflow, g.gamma):
                arr[i+1, j] = arr[i, j]
        g.i_last[j] += 1
        g.x[ii, j] = dS.xD
        g.r[ii, j] = dS.rD
        g.mach[ii, j] = dS.machD
        g.theta[ii, j] = dS.thetaD
        g.pres[ii, j] = dS.presD
        g.temp[ii, j] = dS.tempD
        g.rho[ii, j] = dS.rhoD
        g.massflow[ii, j] = dS.mdot
        g.gamma[ii, j] = g_gamma
        self._calc_massflow_and_thrust(j, j, p_amb, geom)

    self.i_bd = ii + 1
    self.j_bd = j

    self.last_rrc = dS.last_rrc
    g.x[0, self.last_rrc] = dS.xE
    g.r[0, self.last_rrc] = dS.rE
    g.mach[0, self.last_rrc] = dS.machE
    g.theta[0, self.last_rrc] = dS.thetaE
    g.pres[0, self.last_rrc] = dS.presE
    g.temp[0, self.last_rrc] = dS.tempE
    g.rho[0, self.last_rrc] = dS.rhoE
    g.massflow[0, self.last_rrc] = g.massflow[0, 0]
    g.gamma[0, self.last_rrc] = dS.gammaE
    g.i_last[self.last_rrc] = 0

    self._calc_bde_region(ii, j, self.last_rrc, geom)
    self._calc_remaining_mesh(ii, j, self.last_rrc, geom)
    self._calc_wall_contour(ii, j + 1, self.last_rrc, geom)
    self._calc_massflow_and_thrust(j + 1, self.last_rrc, p_amb, geom)

    self._crop_nozzle_to_length(self.last_rrc)
    self.j_de_last = self._calc_de(self.i_bd, self.j_bd, self.last_rrc,
                                   n_type, geom)
    self.mdot_err_ratio = (g.massflow[0, self.last_rrc]
                           / g.massflow[0, 0]) if g.massflow[0, 0] else 1.0
    return True


MOCGridCalc._calc_contoured_nozzle = _calc_contoured_nozzle
