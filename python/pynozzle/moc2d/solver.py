"""Main 2D / axisymmetric MOC solver -- direct port of the C++
``MOC_GridCalc`` class from ``MOC_GridCalc_BDE.cpp``.

Numerical convention notes
--------------------------
* All distances are normalized by the throat radius R*.
* All angles internal to the class are in radians; degrees only appear at
  I/O boundaries.
* Indexing: ``i = 0`` is the wall, ``i = iLast[j]`` is the centerline.
  ``j = 0`` is the initial throat line, ``j`` increases downstream.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..common.constants import (
    AXI,
    CONE,
    DEG_PER_RAD,
    ENDPOINT,
    EPS,
    EXITMACH,
    EXITPRESSURE,
    FIXEDEND,
    GASCON,
    GRAV,
    NOGEOM,
    NOPARAM,
    NOTYPE,
    NOZZLELENGTH,
    PERFECT,
    PI,
    RAD_PER_DEG,
    RAO,
    SEC_FAIL,
    SEC_FAIL_HIGH,
    SEC_FAIL_LOW,
    SEC_OK,
    TWOD,
)
from ..common.thermo import (
    calc_A,
    calc_B,
    calc_MM,
    calc_R,
    calc_R_star,
    calc_b,
    calc_mu,
    isentropic_p_t_rho,
    l_dy_dx,
    r_dy_dx,
    tan_avg,
)
from .grid import MOCGrid


# Sentinel value used in the C++ code to indicate a hard-stop failure inside
# the ARC-walking routines (``return -999019;`` in ``CalcRRCsAlongArc``).
_HARD_FAIL = -999019


@dataclass
class MOCResult:
    """Return value of :meth:`MOCGridCalc.run`.

    Contains the final populated grid plus a handful of summary scalars that
    the I/O writer uses to populate ``summary.out``.
    """
    success: bool
    grid: Optional[MOCGrid] = None
    last_rrc: int = 0
    j_bd: int = 0
    i_bd: int = 0
    j_de_last: int = 0
    theta_b_ans: float = 0.0       # final ThetaB (radians)
    mdot_err_ratio: float = 1.0
    nozzle_type: int = NOTYPE
    nozzle_geom: int = NOGEOM
    design_param: int = NOPARAM
    design_value: tuple = (0.0, 0.0)
    p_total: float = 0.0
    t_total: float = 0.0
    p_ambient: float = 0.0
    mol_wt: float = 0.0
    gamma_i: float = 0.0
    isp_ideal: float = 0.0
    rwt_u: float = 1.0
    rwt_d: float = 1.0
    n_c: int = 0
    dt_limit_rad: float = 0.0
    n_rrc_above_bd: int = 0
    n_sl_i: int = 0
    n_sl_j: int = 0
    print_mode: int = 0
    error_message: str = ""


# ---------------------------------------------------------------------------
class _DEPoint:
    """Minimal port of the relevant fields of the C++ ``dummyStruct`` for
    ``CalcLRCDE`` / ``FindPointE`` return values."""
    __slots__ = (
        "xD", "rD", "machD", "thetaD", "presD", "tempD", "rhoD",
        "xE", "rE", "machE", "thetaE", "presE", "tempE", "rhoE", "gammaE",
        "mdot", "wE", "i", "last_rrc", "status",
    )

    def __init__(self) -> None:
        for s in self.__slots__:
            setattr(self, s, 0.0)
        self.i = 0
        self.last_rrc = 0
        self.status = SEC_OK


# ---------------------------------------------------------------------------
class MOCGridCalc:
    """2D / axisymmetric Method-of-Characteristics nozzle grid calculator.

    Public API mirrors the original C++ class as closely as a Pythonic
    interface allows. Typical usage::

        calc = MOCGridCalc()
        calc.set_initial_properties(...)
        calc.set_solution_parameters(...)
        result = calc.run()
    """

    # ----- construction ------------------------------------------------
    def __init__(self, max_lrc: int = 1000, max_rrc: int = 999) -> None:
        self.grid = MOCGrid(max_lrc, max_rrc)
        self.max_lrc = max_lrc
        self.max_rrc = max_rrc

        # --- defaults copied from the C++ constructor ---
        self.nC: int = -99
        self.RWTD: float = 1.0
        self.RWTU: float = 1.0
        self.DTLIMIT: float = 0.5 * RAD_PER_DEG
        self.nozzle_geom: int = NOGEOM
        self.nozzle_type: int = NOTYPE
        self.design_param: int = NOPARAM
        self.design_value: list[float] = [3.0, -99.9]
        self.con_crit: float = 1e-10
        self.theta_b_min: float = 0.1 * RAD_PER_DEG
        self.theta_b_max: float = 50.0 * RAD_PER_DEG
        self.last_rrc: int = 0
        self.print_mode: int = 0
        self.m_throat: float = -99.9

        # --- placeholders filled in by ``set_initial_properties`` ---
        self.throat_flag: int = 0
        self.mol_wt: float = 0.0
        self.gamma_i: float = 0.0
        self.p_total: float = 0.0
        self.t_total: float = 0.0
        self.p_ambient: float = 0.0
        self.n_rrc_above_bd: int = 0
        self.n_sl_i: int = 0
        self.n_sl_j: int = 0
        self.isp_ideal: float = 1.0
        self.theta_bi: float = 25.0 * RAD_PER_DEG

        # state computed by the solve
        self.i_bd: int = 0
        self.j_bd: int = 0
        self.j_de_last: int = 0
        self.theta_b_ans: float = 0.0
        self.mdot_err_ratio: float = 1.0

    # ----- public configuration ----------------------------------------
    def set_initial_properties(
        self,
        pres: float,
        temp: float,
        mol_wt: float,
        gamma: float,
        p_amb: float,
        n: int,
        rwt_u: float,
        rwt_d: float,
        d_t_limit_deg: float,
        n_rrc_above_bd: int,
        n_sl_i: int,
        n_sl_j: int,
        vel: float,
        throat_flag: int,
        isp_ideal: float,
    ) -> bool:
        """Direct port of ``MOC_GridCalc::SetInitialProperties``.

        Returns ``True`` on success, ``False`` otherwise.
        """
        if pres <= 0.0 or temp <= 0.0 or mol_wt <= 0.0 or gamma <= 0.0:
            return False
        if n % 2 == 0:
            n += 1   # ensure an odd number of starting characteristics

        self.throat_flag = throat_flag
        self.mol_wt = mol_wt
        self.gamma_i = gamma
        self.nC = n
        self.RWTD = rwt_d
        self.RWTU = rwt_u
        self.DTLIMIT = d_t_limit_deg * RAD_PER_DEG
        self.p_ambient = p_amb
        self.n_rrc_above_bd = n_rrc_above_bd
        self.n_sl_i = n_sl_i
        self.n_sl_j = n_sl_j
        self.isp_ideal = isp_ideal
        self.m_throat = vel / math.sqrt(gamma * GASCON / mol_wt * GRAV * temp)

        if throat_flag == 1:
            # input conditions are throat static conditions
            self.t_total = temp * (1.0 + (gamma - 1.0) / 2.0
                                   * self.m_throat * self.m_throat)
            self.p_total = pres * (self.t_total / temp) ** (gamma / (gamma - 1.0))
        else:
            self.p_total = pres
            self.t_total = temp
        return True

    def set_solution_parameters(
        self,
        geom: int,
        nozzle_type: int,
        design_param: int,
        value1: float,
        value2: float = -99.9,
        theta_bi_deg: float = 25.0,
    ) -> None:
        """Direct port of both overloads of ``SetSolutionParameters``."""
        self.nozzle_geom = geom
        self.nozzle_type = nozzle_type
        self.design_param = design_param
        self.design_value = [value1, value2]
        self.theta_bi = theta_bi_deg  # caller passes degrees; will scale later
        if design_param == ENDPOINT:
            self.nozzle_type = FIXEDEND

    def set_print_mode(self, mode: int) -> None:
        self.print_mode = mode

    # ----- short numerical helpers (delegated) --------------------------
    # These are kept as instance methods to mirror the C++ class API.
    @staticmethod
    def CalcMu(mach: float) -> float:
        return calc_mu(mach)

    @staticmethod
    def MM(mach: float) -> float:
        return calc_MM(mach)

    @staticmethod
    def CalcA(mach: float, gamma: float) -> float:
        return calc_A(mach, gamma)

    @staticmethod
    def CalcB(mach: float, theta: float, r: float) -> float:
        return calc_B(mach, theta, r)

    @staticmethod
    def Calcb(mach: float, theta: float, r: float) -> float:
        return calc_b(mach, theta, r)

    @staticmethod
    def CalcR(mach: float, theta: float, r: float) -> float:
        return calc_R(mach, theta, r)

    @staticmethod
    def CalcRStar(mach: float, theta: float, r: float) -> float:
        return calc_R_star(mach, theta, r)

    @staticmethod
    def lDyDx(theta: float, mu: float) -> float:
        return l_dy_dx(theta, mu)

    @staticmethod
    def rDyDx(theta: float, mu: float) -> float:
        return r_dy_dx(theta, mu)

    @staticmethod
    def TanAvg(x: float, y: float) -> float:
        return tan_avg(x, y)

    # ----- isentropic helpers ------------------------------------------
    def calc_isentropic_p_t_rho_scalar(
        self, gamma: float, mach: float
    ) -> tuple[float, float, float]:
        """Equivalent to ``CalcIsentropicP_T_RHO(double, double)``: returns
        (pressure psia, temperature R, density slug/ft^3)."""
        st = isentropic_p_t_rho(
            self.p_total, self.t_total, self.mol_wt, gamma, mach
        )
        return st.pressure, st.temperature, st.density

    def _set_isentropic(self, i: int, j: int, gamma: float, mach: float) -> None:
        """Equivalent to the index-aware overload
        ``CalcIsentropicP_T_RHO(int, int, double, double)``."""
        p, t, rho = self.calc_isentropic_p_t_rho_scalar(gamma, mach)
        g = self.grid
        g.gamma[i, j] = gamma
        g.mach[i, j] = mach
        g.pres[i, j] = p
        g.temp[i, j] = t
        g.rho[i, j] = rho

    # ===================================================================
    # The solver is split across many methods that all share ``self.grid``
    # state.  We're including them in this module rather than splitting
    # across files because every method touches the shared arrays and that
    # is much easier to follow when they all live together (as in the
    # original ``MOC_GridCalc_BDE.cpp``).
    # ===================================================================

    # Body of solver continues in solver_part2.py via mixin -- see end of file.

    def run(self) -> MOCResult:
        """Run the full calculation. Equivalent to ``CreateMOCGrid``."""
        if self.nC < 5:
            return MOCResult(
                success=False,
                error_message=(
                    "Number of starting characteristics must be > 5; got "
                    f"{self.nC}."
                ),
            )
        return self._calc_moc_grid()

    # ----- main controller --------------------------------------------
    def _calc_moc_grid(self) -> MOCResult:
        """Direct port of ``MOC_GridCalc::CalcMOC_Grid``."""
        if self.nozzle_type == NOTYPE:
            return MOCResult(success=False, error_message="Nozzle type not set")
        if self.design_param == NOPARAM:
            return MOCResult(success=False, error_message="Design parameter not set")
        if self.nozzle_geom == NOGEOM:
            return MOCResult(success=False, error_message="Nozzle geometry not set")

        g = self.grid
        g.i_last[0] = self.nC - 1

        try:
            self._calc_initial_throat_line(
                self.RWTU, int(g.i_last[0]), self.gamma_i, self.p_ambient,
                self.nozzle_geom, self.throat_flag, self.m_throat
            )
        except _SecantFail as e:
            return MOCResult(success=False, error_message=str(e))

        try:
            if self.nozzle_type == CONE:
                ok = self._calc_cone_nozzle(
                    self.design_param, self.design_value, self.p_ambient,
                    self.nozzle_geom, self.n_sl_i, self.n_sl_j
                )
            else:
                ok = self._calc_contoured_nozzle(
                    self.design_param, self.design_value, self.gamma_i,
                    self.p_ambient, self.nozzle_geom, self.n_rrc_above_bd,
                    self.nozzle_type, self.n_sl_i, self.n_sl_j,
                )
        except _SecantFail as e:
            return MOCResult(success=False, error_message=str(e))

        if not ok:
            return MOCResult(success=False, error_message="Solver did not converge")

        return MOCResult(
            success=True,
            grid=g,
            last_rrc=self.last_rrc,
            j_bd=self.j_bd,
            i_bd=self.i_bd,
            j_de_last=self.j_de_last,
            theta_b_ans=self.theta_b_ans,
            mdot_err_ratio=self.mdot_err_ratio,
            nozzle_type=self.nozzle_type,
            nozzle_geom=self.nozzle_geom,
            design_param=self.design_param,
            design_value=tuple(self.design_value),
            p_total=self.p_total,
            t_total=self.t_total,
            p_ambient=self.p_ambient,
            mol_wt=self.mol_wt,
            gamma_i=self.gamma_i,
            isp_ideal=self.isp_ideal,
            rwt_u=self.RWTU,
            rwt_d=self.RWTD,
            n_c=self.nC,
            dt_limit_rad=self.DTLIMIT,
            n_rrc_above_bd=self.n_rrc_above_bd,
            n_sl_i=self.n_sl_i,
            n_sl_j=self.n_sl_j,
            print_mode=self.print_mode,
        )


class _SecantFail(RuntimeError):
    """Internal exception used to abort the solve when a secant-method loop
    cannot converge. Mirrors the C++ ``AfxMessageBox; exit(1)`` pattern but
    in a recoverable way."""


# The rest of the solver methods are attached below via simple monkey-patching;
# we do it this way to keep file lengths manageable without sacrificing the
# "one big class, like in the C++" mental model.
from . import _solver_kernels  # noqa: E402,F401
from . import _contoured_kernels  # noqa: E402,F401
