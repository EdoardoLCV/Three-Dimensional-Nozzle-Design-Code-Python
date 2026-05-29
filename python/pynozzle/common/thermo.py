"""Thermodynamic helper functions used by all three nozzle tools.

These wrap the small math kernels that show up in many places in the original
C++ code: Mach angle, Prandtl-Meyer function, isentropic property ratios, etc.
Keeping them in one module avoids drift between the three tools.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import GASCON, GRAV, PI


# --- small kernels (direct ports of MOC_GridCalc helpers) -------------------
def calc_mu(mach: float) -> float:
    """Mach angle ``mu = asin(1/M)`` (radians).

    Returns NaN when ``mach < 1.0`` -- mirrors the C++ behaviour where
    the iterative loops detect this case via the surrounding ``M[3] >=
    1.0`` guards. Python's :func:`math.asin` would raise instead.
    """
    if mach < 1.0:
        return float("nan")
    return math.asin(1.0 / mach)


def calc_MM(mach: float) -> float:
    """Return ``sqrt(M^2 - 1)`` (the ``MM`` helper in the original).

    The C++ code allows this to silently produce ``NaN`` when the Mach
    number dips below 1.0 mid-iteration; the surrounding iterative loops
    detect non-convergence via ``M[3] >= 1.0`` and bail out. Python's
    :func:`math.sqrt` would raise ``ValueError`` instead, so we return
    ``NaN`` explicitly to preserve the C++ semantics.
    """
    arg = mach * mach - 1.0
    if arg < 0.0:
        return float("nan")
    return math.sqrt(arg)


def calc_A(mach: float, gamma: float) -> float:
    """First term of the dtheta equation (RAO eq. 15)."""
    mm = calc_MM(mach)
    denom = mach * (1.0 + (gamma - 1.0) / 2.0 * mach * mach)
    if denom == 0.0:
        return float("nan")
    return mm / denom


def calc_B(mach: float, theta: float, r: float) -> float:
    """Second term of the dtheta equation for a left-running characteristic.

    The C++ relies on ``tan(0) = 0`` producing ``inf`` for the inner
    division and then ``1/(r*inf) = 0``. Python raises ZeroDivisionError
    for ``1/0``, so we short-circuit theta=0 to return 0 explicitly.
    """
    if r == 0.0 or theta == 0.0:
        return 0.0
    return 1.0 / (r * (calc_MM(mach) / math.tan(theta) - 1.0))


def calc_b(mach: float, theta: float, r: float) -> float:
    """Second term of the dtheta equation for a right-running characteristic."""
    if r == 0.0 or theta == 0.0:
        return 0.0
    return 1.0 / (r * (calc_MM(mach) / math.tan(theta) + 1.0))


def calc_R(mach: float, theta: float, r: float) -> float:
    """``R`` helper for the LRC dtheta integration."""
    if r == 0.0:
        return 0.0
    if theta == 0.0:
        # Equivalent to the C++ ``1/(r*(MM + 1/tan(0)))``: 1/tan(0)=inf,
        # so the result is 0.
        return 0.0
    return 1.0 / (r * (calc_MM(mach) + 1.0 / math.tan(theta)))


def calc_R_star(mach: float, theta: float, r: float) -> float:
    """``R*`` helper for the LRC dtheta integration."""
    if r == 0.0:
        return 0.0
    if theta == 0.0:
        return 0.0
    return 1.0 / (r * (calc_MM(mach) - 1.0 / math.tan(theta)))


def l_dy_dx(theta: float, mu: float) -> float:
    """Slope of an LRC characteristic line."""
    return math.tan(theta + mu)


def r_dy_dx(theta: float, mu: float) -> float:
    """Slope of an RRC characteristic line."""
    return math.tan(theta - mu)


def tan_avg(x: float, y: float) -> float:
    """``tan( (atan(x) + atan(y)) / 2 )`` -- the "tangent-averaging" helper."""
    return math.tan(0.5 * (math.atan(x) + math.atan(y)))


# --- Prandtl-Meyer ----------------------------------------------------------
def prandtl_meyer(mach: float, gamma: float) -> float:
    """Prandtl-Meyer function in radians (Anderson, MCF, p. 368)."""
    g = gamma
    return (
        math.sqrt((g + 1.0) / (g - 1.0))
        * math.atan(math.sqrt((g - 1.0) / (g + 1.0) * (mach * mach - 1.0)))
        - math.atan(math.sqrt(mach * mach - 1.0))
    )


# --- isentropic state -------------------------------------------------------
@dataclass
class IsentropicState:
    """Container for static pressure, temperature, and density at a point.

    Pressure is in psia, temperature in °R, density in slug/ft³ -- matching the
    units used by the original C++ code's ``CalcIsentropicP_T_RHO``.
    """
    pressure: float
    temperature: float
    density: float


def isentropic_p_t_rho(
    p_total: float,
    t_total: float,
    mol_wt: float,
    gamma: float,
    mach: float,
) -> IsentropicState:
    """Compute static p, T, rho from stagnation conditions and Mach number.

    Direct port of ``MOC_GridCalc::CalcIsentropicP_T_RHO(double, double)``.

    Parameters
    ----------
    p_total
        Total (stagnation) pressure, psia.
    t_total
        Total (stagnation) temperature, °R.
    mol_wt
        Molecular weight, lbm/lbm-mol.
    gamma
        Ratio of specific heats.
    mach
        Local Mach number.
    """
    ratio = 1.0 + (gamma - 1.0) / 2.0 * mach * mach
    p = p_total / ratio ** (gamma / (gamma - 1.0))             # psia
    t = t_total / ratio                                        # °R
    # density: slug/ft^3.  p[psia]*144 = lbf/ft^2; R_specific = GASCON/mol_wt;
    # rho [slug/ft^3] = p / (R_specific * T * GRAV)
    rho = (p * 144.0) / (GASCON / mol_wt * t * GRAV)
    return IsentropicState(p, t, rho)


# --- speed of sound / static velocity --------------------------------------
def speed_of_sound(temperature: float, mol_wt: float, gamma: float) -> float:
    """Sound speed (ft/s) given temperature (°R), molecular weight, gamma."""
    return math.sqrt(gamma * GASCON / mol_wt * GRAV * temperature)


def velocity_from_mach(
    mach: float, temperature: float, mol_wt: float, gamma: float
) -> float:
    """Static flow speed (ft/s) at given Mach number and temperature."""
    return mach * speed_of_sound(temperature, mol_wt, gamma)


# --- choked flow at the throat (1-D) ----------------------------------------
def one_d_throat_mdot_per_area(
    p_total: float, mol_wt: float, gamma: float
) -> float:
    """Choked 1-D mass-flow rate per unit area at the throat, lbm/(s·in²).

    Used by the summary file to compare 2-D MOC mass flow against the 1-D
    isentropic value (the discharge coefficient C_D).
    """
    # Throat static pressure for M=1 from isentropic relations:
    p_star = p_total / (1.0 + (gamma - 1.0) / 2.0) ** (gamma / (gamma - 1.0))
    t_star = 1.0 / (1.0 + (gamma - 1.0) / 2.0)  # T*/T_total (dimensionless)
    # Reproduce the same algebra as CalcIsentropicP_T_RHO + the OutputSummaryFile:
    return GRAV * p_star * math.sqrt(gamma / (GASCON * GRAV / mol_wt * t_star))


__all__ = [
    "IsentropicState",
    "calc_A",
    "calc_B",
    "calc_MM",
    "calc_R",
    "calc_R_star",
    "calc_b",
    "calc_mu",
    "isentropic_p_t_rho",
    "l_dy_dx",
    "one_d_throat_mdot_per_area",
    "prandtl_meyer",
    "r_dy_dx",
    "speed_of_sound",
    "tan_avg",
    "velocity_from_mach",
]
