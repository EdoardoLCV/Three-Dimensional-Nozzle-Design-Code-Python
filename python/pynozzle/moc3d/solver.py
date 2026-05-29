"""3D Method-of-Characteristics solver -- port of ``C3D_MOCGrid``.

The class marches a 3D MOC solution plane-by-plane down a nozzle whose
wall is given as a stack of circles (see :mod:`pynozzle.moc3d.geo`).
Structure and variable names track the C++ closely; the Numerical
Recipes routines are replaced with NumPy / SciPy:

* ``ludcmp`` / ``lubksb``     -> ``numpy.linalg.solve`` / SciPy LU
* ``newt`` (Newton-Raphson)   -> ``scipy.optimize.fsolve`` with analytic Jacobian
* ``sort2``                   -> ``numpy.argsort``

The thin-plate-spline surface fit and the per-point property evaluation
are vectorised for speed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy.linalg import lu_factor, lu_solve
    from scipy.optimize import fsolve
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

from .point import (
    XYZPoint, PI, DEG_PER_RAD, RAD_PER_DEG, GRAV, GASCON,
    CONSTANT_X, CONSTANT_Y, LINE, CIRCLE,
    FAIL, FAIL_MACH, FAIL_HIGH, OK,
)
from .geo import GeoConfig


@dataclass
class MOC3DResult:
    """Result of a :meth:`MOC3DGrid.calc_nozzle` run."""
    success: bool
    n_pts: int = 0
    n_z: int = 0
    n_div: int = 0
    output_dir: Optional[str] = None
    error_message: str = ""


class MOC3DGrid:
    """Port of the ``C3D_MOCGrid`` class."""

    def __init__(self, cfg: GeoConfig, output_dir: str | Path = "."):
        self.cfg = cfg
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._print_mode = 1
        self._x_step = cfg.x_output_step
        self._y_step = cfg.y_output_step
        self._z_step = cfg.z_output_step
        self._step_step = cfg.step_step
        self.s_fit = cfg.surface_fit

        self._n_div = cfg.n_div
        self._n_z = cfg.n_z
        self._n_pts = 0
        self._n_radii = 0
        self._n_neighbors = 8

        # Point arrays (lists of lists of XYZPoint)
        self._pt: list[list[XYZPoint]] = []
        self._neighbor: list[list[XYZPoint]] = []
        self._wall_pt: list[list[XYZPoint]] = []
        self._body_point_flag: list[bool] = []

        # Surface-fit coefficients (filled per plane)
        self._cfit = np.zeros((1, 7))
        # cached plane arrays for vectorised surface evaluation
        self._fit_x = np.zeros(0)
        self._fit_y = np.zeros(0)

        # Newton-Raphson coefficient matrix (global ``_a`` in the C++)
        self._a = np.zeros((3, 4))

        self._outfile = None

    # ==================================================================
    #  small math helpers (mirror the C++ inline helpers)
    # ==================================================================
    @staticmethod
    def _calc_mu(mach: float) -> float:
        if mach < 1.0:
            if mach < 0.999:
                raise RuntimeError("Could not calculate Mach angle for M < 1.0")
            return PI / 2.0
        return math.asin(1.0 / mach)

    # ==================================================================
    #  geometry setup
    # ==================================================================
    def _initialize_wall_points(self):
        cols = self._n_z * 5
        self._wall_pt = [[XYZPoint() for _ in range(cols)]
                         for _ in range(self._n_div)]

    def _initialize_data_members(self):
        cols = self._n_z * 5
        self._neighbor = [[XYZPoint() for _ in range(9)]
                          for _ in range(self._n_pts)]
        self._pt = [[XYZPoint() for _ in range(cols)]
                    for _ in range(self._n_pts)]
        self._body_point_flag = [False] * self._n_pts

    def set_initial_properties(self) -> bool:
        """Port of ``SetInitialPropertiesForCircularThroat`` (geometry part).

        Reads the stacked-circle wall from the config, builds the wall
        point array, writes ``Initial Wall.plt`` and seeds the initial
        reference plane (k=0).
        """
        from .io_writers import _fmt
        cfg = self.cfg
        if (cfg.p0 <= 0.0 or cfg.t0 <= 0.0 or cfg.mol_wt0 <= 0.0
                or cfg.gamma0 <= 0.0 or cfg.mach0 < 1.0):
            raise ValueError("All input parameters must be > 0 and Mach >= 1")

        self._n_z = cfg.n_z
        self._n_div = cfg.n_div
        self._initialize_wall_points()

        wall_path = self.out_dir / "Initial Wall.plt"
        with open(wall_path, "w") as f:
            f.write('VARIABLES = "X(in)","Y(in)","Z(in)"\n')
            f.write('TITLE = "Initial Wall Plot"\n')
            f.write('text  x=5  y=93  t="Initial Wall Plot"\n')
            for k in range(self._n_z):
                f.write(f'zone t="K = {k}" I ={self._n_div + 1} J = 1 K = 1\n')
                z0 = cfg.z[k]
                r0 = cfg.r[k]
                x0 = cfg.xc[k]
                y0 = cfg.yc[k]
                for j in range(self._n_div):
                    wp = self._wall_pt[j][k]
                    wp.z = z0
                    angle = j * 2 * PI / self._n_div
                    wp.x = r0 * math.cos(angle) + x0
                    wp.y = r0 * math.sin(angle) + y0
                    if abs(wp.x) < 1e-10:
                        wp.x = 0.0
                    if abs(wp.y) < 1e-10:
                        wp.y = 0.0
                    if k == 0 and j == 0:
                        self._set_initial_reference_plane(
                            cfg.p0, cfg.t0, cfg.mach0, cfg.mol_wt0, cfg.gamma0,
                            cfg.theta0, cfg.psi0, z0, r0, self._n_div)
                    f.write(f"{_fmt(wp.x)}\t{_fmt(wp.y)}\t{_fmt(wp.z)}\n")
                wp0 = self._wall_pt[0][k]
                f.write(f"{_fmt(wp0.x)}\t{_fmt(wp0.y)}\t{_fmt(wp0.z)}\n\n")
        return True

    def _set_initial_reference_plane(self, pres, temp, mach, m_wt, gamma,
                                     theta, psi, z0, r0, n_div):
        """Port of ``SetInitialReferencePlane`` -- seeds plane k=0 and
        writes ``z=0.out``."""
        from .io_writers import _fmt
        dis = 2 * PI * r0 / n_div
        self._n_radii = int(r0 / dis) + 2
        rad_step = r0 / self._n_radii

        # First pass: count points
        k = 0
        for j in range(self._n_radii + 1):
            rad = j * rad_step
            n_pt = int(2 * PI * rad / rad_step)
            if n_pt < 7 and rad > 0.0:
                n_pt = 7
            for _ in range(n_pt + 1):
                k += 1
        self._n_pts = k
        self._initialize_data_members()

        rho = pres * 144 / (GASCON / m_wt * temp)
        q = mach * math.sqrt(gamma * GASCON / m_wt * temp * GRAV)

        path = self.out_dir / "z=0.out"
        with open(path, "w") as f:
            # NOTE: the C++ source has a "Radius(in)" column here, but the
            # upstream sample outputs were generated by an earlier version
            # that didn't write it. We match the SAMPLE format for
            # validation.
            f.write("X(in)\tY(in)\tZ(in)\tP(psia)\tT(R)\t"
                    "RHO(lbm/ft3)\tMach\tVelocity(ft/s)\tTheta(deg)\tPsi(deg)\t"
                    "Radii#\tPt#\tTotal Pt#\tBodyFlag\n")
            rad_step = r0 / self._n_radii
            k = 0
            for j in range(self._n_radii + 1):
                rad = j * rad_step
                n_pt = int(2 * PI * rad / rad_step)
                if n_pt < 7 and rad > 0.0:
                    n_pt = 7
                for i in range(n_pt + 1):
                    pt = self._pt[k][0]
                    pt.x = rad * math.cos(2 * PI * i / (n_pt + 1))
                    pt.y = rad * math.sin(2 * PI * i / (n_pt + 1))
                    pt.z = z0
                    pt.p = pres
                    pt.g = gamma
                    pt.mach = mach
                    pt.mol_wt = m_wt
                    pt.t = temp
                    pt.rho = rho
                    pt.theta = theta * RAD_PER_DEG
                    pt.psi = psi * RAD_PER_DEG
                    pt.q = q
                    self._body_point_flag[k] = (j == self._n_radii)
                    f.write(
                        f"{_fmt(pt.x)}\t{_fmt(pt.y)}\t{_fmt(pt.z)}\t"
                        f"{_fmt(pt.p)}\t{_fmt(pt.t)}\t"
                        f"{_fmt(pt.rho)}\t{_fmt(pt.mach)}\t{_fmt(pt.q)}\t"
                        f"{_fmt(pt.theta*DEG_PER_RAD)}\t"
                        f"{_fmt(pt.psi*DEG_PER_RAD)}\t{j}\t{i}\t{k}\t"
                        f"{int(self._body_point_flag[k])}\n")
                    k += 1
        return True

    # ==================================================================
    #  main driver
    # ==================================================================
    def calc_nozzle(self) -> MOC3DResult:
        """Port of ``CalcNozzle`` -- the plane-by-plane MOC march."""
        self._outfile = open(self.out_dir / "outfile.out", "w")
        try:
            timer = 0
            fail_counter = 0
            self._n_neighbors = 8
            k = 1
            while k < self._n_z:
                if self.s_fit == "All Point Spline":
                    self._all_point_surface_fit(k - 1)
                else:
                    self._set_neighbor_points(k - 1, k - 1)

                i = 0
                while i < self._n_pts:
                    dz = self._wall_pt[0][k].z - self._wall_pt[0][k - 1].z
                    if not self._body_point_flag[i]:
                        if not self._calc_field_point(i, k - 1, dz) and fail_counter < 15:
                            fail_counter += 1
                            self._add_new_nozzle_point(k)
                            i = -1
                    else:
                        cbp = self._calc_body_point(i, k - 1, dz)
                        if cbp == FAIL_MACH and fail_counter < 15:
                            fail_counter += 1
                            self._add_new_nozzle_point(k)
                            i = -1
                        elif fail_counter >= 15 or cbp == FAIL:
                            return MOC3DResult(
                                success=False, n_pts=self._n_pts, n_z=self._n_z,
                                n_div=self._n_div, error_message="Body Point Calculation FAILED")
                    i += 1

                timer += 1
                if timer == self._step_step:
                    self._output_contour(0, self._n_pts - 1, 0, k)
                    timer = 0
                k += 1
            self._output_contour(0, self._n_pts - 1, 0, k - 1)
        finally:
            if self._outfile:
                self._outfile.close()
        return MOC3DResult(success=True, n_pts=self._n_pts, n_z=self._n_z,
                           n_div=self._n_div, output_dir=str(self.out_dir))

    def _add_new_nozzle_point(self, k: int):
        """Port of ``AddNewNozzlePoint`` -- inserts a wall plane at k."""
        self._n_z += 1
        # arrays were allocated with _n_z*5 columns; ensure capacity
        need = self._n_z + 1
        for j in range(self._n_div):
            while len(self._wall_pt[j]) < need:
                self._wall_pt[j].append(XYZPoint())
        for i in range(self._n_pts):
            while len(self._pt[i]) < need:
                self._pt[i].append(XYZPoint())

        for i in range(self._n_z, k, -1):
            for j in range(self._n_div):
                self._wall_pt[j][i].x = self._wall_pt[j][i - 1].x
                self._wall_pt[j][i].y = self._wall_pt[j][i - 1].y
                self._wall_pt[j][i].z = self._wall_pt[j][i - 1].z
        for j in range(self._n_div):
            self._wall_pt[j][k].x = (self._wall_pt[j][k - 1].x + self._wall_pt[j][k + 1].x) / 2.0
            self._wall_pt[j][k].y = (self._wall_pt[j][k - 1].y + self._wall_pt[j][k + 1].y) / 2.0
            self._wall_pt[j][k].z = (self._wall_pt[j][k - 1].z + self._wall_pt[j][k + 1].z) / 2.0

    # ------------------------------------------------------------------
    #  surface fit
    # ------------------------------------------------------------------
    def _all_point_surface_fit(self, k: int):
        """Port of ``AllPointSurfaceFit`` (thin-plate spline over all points)."""
        n = self._n_pts
        x = np.array([self._pt[i][k].x for i in range(n)])
        y = np.array([self._pt[i][k].y for i in range(n)])
        props = np.empty((n, 7))
        for i in range(n):
            p = self._pt[i][k]
            props[i] = (p.p, p.rho, p.q, p.theta, p.psi, p.g, p.mol_wt)

        # Build the (n+3) x (n+3) TPS matrix
        A = np.zeros((n + 3, n + 3))
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        r = dx * dx + dy * dy
        with np.errstate(divide="ignore", invalid="ignore"):
            K = np.where(r != 0.0, r * np.log(r), 0.0)
        A[:n, 0] = 1.0
        A[:n, 1] = x
        A[:n, 2] = y
        A[:n, 3:] = K
        A[n, 3:] = y
        A[n + 1, 3:] = x
        A[n + 2, 3:] = 1.0

        b = np.zeros((n + 3, 7))
        b[:n, :] = props

        if _HAVE_SCIPY:
            lu, piv = lu_factor(A)
            sol = lu_solve((lu, piv), b)
        else:  # pragma: no cover
            sol = np.linalg.solve(A, b)

        self._cfit = sol  # shape (n+3, 7)
        self._fit_x = x
        self._fit_y = y

    def _calc_surface_point_properties(self, base: XYZPoint, i: int, k: int):
        """Port of ``CalcSurfacePointProperties`` (All-Point branch).

        Updates ``base``'s flow state from the surface fit and returns the
        derivative dict with keys 3,4,6,8 (dtdx, dpdx, dtdy, dpdy).
        """
        cfit = self._cfit
        x = self._fit_x
        y = self._fit_y

        dx = base.x - x
        dy = base.y - y
        rad = dx * dx + dy * dy
        mask = (rad > 0.0) & np.isfinite(rad)
        rlogr = np.zeros_like(rad)
        rlogr[mask] = rad[mask] * np.log(rad[mask])

        # val[ival] = cfit[0]+cfit[1]*x+cfit[2]*y + sum_j cfit[j+3]*rlogr_j
        base_poly = cfit[0] + cfit[1] * base.x + cfit[2] * base.y  # (7,)
        rbf = rlogr @ cfit[3:]  # (7,)
        val = base_poly + rbf

        base.p = val[0]
        base.rho = val[1]
        base.q = val[2]
        base.theta = val[3]
        base.psi = val[4]
        base.g = val[5]
        base.mol_wt = val[6]
        if abs(base.theta) < 1e-8:
            base.theta = 0.0
        if abs(base.psi) < 1e-8:
            base.psi = 0.0
        base.t = 144 * base.p / (base.rho * GASCON / base.mol_wt)
        base.mach = base.q / math.sqrt(GRAV * base.g * GASCON / base.mol_wt * base.t)

        # derivatives
        fac = np.zeros_like(rad)
        fac[mask] = 2 * (1 + np.log(rad[mask]))
        dtdx = cfit[1, 3] + np.sum(cfit[3:, 3] * fac * dx)
        dtdy = cfit[2, 3] + np.sum(cfit[3:, 3] * fac * dy)
        dpdx = cfit[1, 4] + np.sum(cfit[3:, 4] * fac * dx)
        dpdy = cfit[2, 4] + np.sum(cfit[3:, 4] * fac * dy)
        return {3: dtdx, 6: dtdy, 4: dpdx, 8: dpdy}

    def _set_neighbor_points(self, k_set: int, k_now: int):
        """Port of ``SetNeighborPoints`` (used by the 9-point fit path)."""
        n = self._n_pts
        x = np.array([self._pt[i][k_set].x for i in range(n)])
        y = np.array([self._pt[i][k_set].y for i in range(n)])
        for i in range(n):
            d = (x[i] - x) ** 2 + (y[i] - y) ** 2
            order = np.argsort(d, kind="stable")
            for kpt in range(self._n_neighbors + 1):
                self._neighbor[i][kpt].copy_from(self._pt[int(order[kpt])][k_now])

    # ------------------------------------------------------------------
    #  field point
    # ------------------------------------------------------------------
    def _calc_field_point(self, i: int, k: int, dz: float) -> bool:
        from .kernels import calc_field_point
        return calc_field_point(self, i, k, dz)

    def _calc_body_point(self, i: int, k: int, dz: float) -> int:
        from .kernels import calc_body_point
        return calc_body_point(self, i, k, dz)

    # ------------------------------------------------------------------
    #  output
    # ------------------------------------------------------------------
    def _output_contour(self, i_start, i_end, k_start, k_end):
        from . import io_writers
        io_writers.output_contour(self, i_start, i_end, k_start, k_end)
