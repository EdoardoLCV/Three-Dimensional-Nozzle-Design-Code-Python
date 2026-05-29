"""Computational kernels for the 3D MOC solver.

These are free functions taking the :class:`MOC3DGrid` instance as their
first argument (mirroring the C++ member functions). Kept separate from
:mod:`solver` to keep the file sizes manageable.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

try:
    from scipy.optimize import fsolve
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

from .point import (
    XYZPoint, PI, GRAV, GASCON,
    CONSTANT_X, CONSTANT_Y, LINE, CIRCLE, FAIL, FAIL_MACH, OK,
)

if TYPE_CHECKING:
    from .solver import MOC3DGrid


def _calc_mu(mach: float) -> float:
    if mach < 1.0:
        if mach < 0.999:
            raise RuntimeError("Could not calculate Mach angle for M < 1.0")
        return PI / 2.0
    return math.asin(1.0 / mach)


# ---------------------------------------------------------------------------
#  Compatibility-equation coefficient (CompEqu)
# ---------------------------------------------------------------------------
def comp_equ(beta, base: XYZPoint, delta, dzdN, dz, theta, dtdN, dpdN):
    """Port of ``CompEqu`` -- returns (a0, a1, a2, rhs)."""
    a0 = 144.0 / (math.tan(beta) * base.rho * base.q * base.q / GRAV)
    a1 = math.cos(delta) - math.sin(beta) * math.sin(delta) * dzdN * base.L / dz
    a2 = math.cos(theta) * (math.sin(delta)
                            + (math.sin(beta) * math.cos(delta) * dzdN * base.L) / dz)
    rhs = (math.sin(beta) * base.L * (math.sin(delta) * dtdN
                                      - math.cos(theta) * math.cos(delta) * dpdN)
           + a0 * base.p + a1 * base.theta + a2 * base.psi)
    return a0, a1, a2, rhs


def calc_parametric_angle(beta, theta, psi, delta):
    """Port of ``CalcParametricAngle`` (the goto-laden delta solver)."""
    a = math.sin(theta) * math.cos(psi) * math.sin(beta)
    b = math.sin(psi) * math.sin(beta)
    c = -0.5 * math.cos(theta) * math.cos(psi) * (math.sin(beta) + math.cos(beta))
    sind = math.sin(delta)
    cosd = math.cos(delta)
    delta_i = delta
    asbs = a * a + b * b
    rad = asbs - c * c
    if rad >= 0.0 and asbs != 0.0:
        rad = math.sqrt(rad) / asbs
        sini = b * c / asbs
        sini1 = sini + a * rad
        sini2 = sini - a * rad
        # The C++ uses goto labels x21..x24; replicate the control flow.
        use_x21 = abs(sind - sini2) < abs(sind - sini1)
        if not use_x21:
            sini = sini2
            if abs(sini) <= 1.0:
                pass  # goto x22
            else:
                use_x21 = True
        if use_x21:
            sini = sini2
            if abs(sini) > 1.0:
                sini = sini1
            if abs(sini) > 1.0:
                return delta_i
        # x22
        cosi = a * c / asbs
        cosi1 = cosi + b * rad
        cosi2 = cosi - b * rad
        use_x23 = abs(cosd - cosi2) < abs(cosd - cosi1)
        if not use_x23:
            cosi = cosi1
            if abs(cosi) <= 1.0:
                pass  # goto x24
            else:
                use_x23 = True
        if use_x23:
            cosi = cosi2
            if abs(cosi) > 1.0:
                cosi = cosi1
            if abs(cosi) > 1.0:
                return delta_i
        # x24
        delta_i = math.atan2(sini, cosi)
        if delta_i < 0.0:
            delta_i += 2 * PI
    return delta_i


# ---------------------------------------------------------------------------
#  Body geometry: BodyFit, FindClosestBodyPoint, etc.
# ---------------------------------------------------------------------------
def body_fit_jk(grid: "MOC3DGrid", j: int, k: int):
    """Port of ``BodyFit(int j, int k)`` -- returns dict with keys
    'type' and 'A','B','C'."""
    n_div = grid._n_div
    jm1 = j - 1
    jp1 = j + 1
    if jm1 == -1:
        jm1 = n_div - 1
    if jp1 == n_div:
        jp1 = 1
    wp = grid._wall_pt
    x = [wp[jm1][k].x, wp[j][k].x, wp[jp1][k].x]
    y = [wp[jm1][k].y, wp[j][k].y, wp[jp1][k].y]

    if x[0] == x[1] and x[0] == x[2]:
        return {"A": 1.0, "B": 0.0, "C": x[1], "type": CONSTANT_X}
    if y[0] == y[1] and y[0] == y[2]:
        return {"A": 0.0, "B": 1.0, "C": y[1], "type": CONSTANT_Y}

    rad = [x[0] * x[0] + y[0] * y[0],
           x[1] * x[1] + y[1] * y[1],
           x[2] * x[2] + y[2] * y[2]]
    dx10 = 2.0 * (x[1] - x[0])
    dy10 = 2.0 * (y[1] - y[0])
    dx21 = 2.0 * (x[2] - x[1])
    dy21 = 2.0 * (y[2] - y[1])
    if dx10 == 0.0:
        dx10 = 2.0 * (x[2] - x[0])
        dx21 = -dx21
        dy10 = 2.0 * (y[2] - y[0])
        dy21 = -dy21
        rad[1], rad[2] = rad[2], rad[1]

    if dx21 != 0.0 and dy10 / dx10 == dy21 / dx21:
        a = -dy10 / dx10
        ds = {"A": a, "B": 1.0, "C": y[0] + x[0] * a, "type": LINE}
    else:
        a = (rad[1] - rad[0]) / dx10
        b = rad[2] - rad[1]
        c = dy10 / dx10
        h = dy21
        if dx21 != 0.0:
            b = b / dx21 - a
            h = h / dx21 - c
        b /= h
        a -= b * c
        c = rad[0] - 2 * (a * x[0] - b * y[0])
        c += b * b + a * a
        ds = {"A": a, "B": b, "C": c, "type": CIRCLE}

    if abs(ds["A"]) < 1e-10:
        ds["A"] = 0.0
    if abs(ds["B"]) < 1e-10:
        ds["B"] = 0.0
    return ds


def find_closest_body_point(grid: "MOC3DGrid", X: XYZPoint):
    """Port of ``FindClosestBodyPoint`` -- returns (j, k)."""
    n_div = grid._n_div
    n_z = grid._n_z
    wp = grid._wall_pt
    k = 0
    while k < n_z and wp[0][k].z < X.z:
        k += 1
    if k >= n_z:
        k = n_z - 1
    if k > 0 and wp[0][k].z - X.z > X.z - wp[0][k - 1].z:
        k -= 1
    min_d = 9e9
    j_min = 0
    for j in range(n_div):
        d = (X.x - wp[j][k].x) ** 2 + (X.y - wp[j][k].y) ** 2
        if d < min_d:
            j_min = j
            min_d = d
    return j_min, k


def body_fit_X(grid: "MOC3DGrid", X: XYZPoint):
    """Port of ``BodyFit(XYZPoint X)``."""
    j, k = find_closest_body_point(grid, X)
    return body_fit_jk(grid, j, k)


def solve_for_body_point_position(CC, bfit, p2x, p2y):
    """Port of ``SolveForBodyPointPosition`` -- returns (x, y)."""
    a = CC["A"]; b = CC["B"]; c = CC["C"]
    a1 = bfit["A"]; b1 = bfit["B"]; c1 = bfit["C"]
    t = bfit["type"]
    if t == CONSTANT_X:
        x = c1
        y = (c - a * x) / b
    elif t == CONSTANT_Y:
        y = c1
        x = (c - b * y) / a
    elif t == LINE:
        y = (a * c1 - c * a1) / (a * b1 - b * a1)
        x = (c - b * y) / a
    else:  # CIRCLE
        if abs(b) <= abs(a):
            d1 = 1 + (b / a) * (b / a)
            d2 = c / a - a1
            d3 = (d2 * d2 + b1 * b1 - c1) / d1
            d2 = (b * d2 / a + b1) / d1
            rad = d2 * d2 - d3
            if rad < 0.0:
                raise RuntimeError("Error in SolveForBodyPointPosition(1)")
            rad = math.sqrt(rad)
            y = d2 + rad
            if abs(p2y - y) >= abs(p2y - d2 + rad):
                y = d2 - rad
            x = (c - b * y) / a
        else:
            d1 = 1 + (a / b) * (a / b)
            d2 = c / b - b1
            d3 = (d2 * d2 + a1 * a1 - c1) / d1
            d2 = (a * d2 / b + a1) / d1
            rad = d2 * d2 - d3
            if rad < 0.0:
                raise RuntimeError("Error in SolveForBodyPointPosition(2)")
            rad = math.sqrt(rad)
            x = d2 + rad
            if abs(p2x - x) > abs(p2x - d2 + rad):
                x = d2 - rad
            y = (c - a * x) / b
    if -1e-5 < x < 1e-5:
        x = 0.0
    if -1e-5 < y < 1e-5:
        y = 0.0
    return x, y


def check_new_point_position(grid: "MOC3DGrid", P: XYZPoint, j_point: int, k: int):
    """Port of ``CheckNewPointPosition`` -- returns (j, k)."""
    n_div = grid._n_div
    wp = grid._wall_pt
    j = j_point
    d_min = (P.x - wp[j][k].x) ** 2 + (P.y - wp[j][k].y) ** 2
    j = j_point - 1
    if j == -1:
        j = n_div - 2
    dis = (P.x - wp[j][k].x) ** 2 + (P.y - wp[j][k].y) ** 2
    if dis < d_min:
        return find_closest_body_point(grid, P)
    j = j_point + 1
    if j == n_div:
        j = 1
    dis = (P.x - wp[j][k].x) ** 2 + (P.y - wp[j][k].y) ** 2
    if dis < d_min:
        return find_closest_body_point(grid, P)
    return j_point, k


def calc_unit_normal_to_body_surface(grid: "MOC3DGrid", P: XYZPoint):
    """Port of ``CalcUnitNormalToBodySurface`` (the active, goto-based body).

    Returns dict {0:n1, 1:n2, 2:n3, 'j':j, 'k':k} where n1 is the Z
    component (positive => diverging), n2 the X, n3 the Y component.
    """
    n_div = grid._n_div
    n_z = grid._n_z
    wp = grid._wall_pt

    j, k = find_closest_body_point(grid, P)
    if k == 0:
        k = 1
    if j == n_div - 1:
        j = n_div - 2

    bf1 = body_fit_jk(grid, j, k - 1)
    a1, b1, c1 = bf1["A"], bf1["B"], bf1["C"]
    bf2 = body_fit_jk(grid, j, k)
    a2, b2, c2 = bf2["A"], bf2["B"], bf2["C"]
    if k == n_z - 1:
        bf3 = bf2
    else:
        bf3 = body_fit_jk(grid, j, k + 1)
    a3, b3, c3 = bf3["A"], bf3["B"], bf3["C"]

    result = {"j": j, "k": k}

    # ---- decide the control-flow branch (mirrors the goto logic) -----
    branch_quadratic = False  # continue24 path (CIRCLE quadratic-in-z)
    branch_general = False    # continue27 path (mixed shapes)
    if bf2["type"] == CIRCLE:
        # continue23
        if bf1["type"] != bf2["type"] or bf3["type"] != bf2["type"]:
            branch_general = True
        else:
            branch_quadratic = True
    else:
        if bf1["type"] <= bf2["type"] and bf3["type"] <= bf2["type"]:
            branch_quadratic = True
        else:
            branch_general = True

    def _finish(n1, n2, n3):
        d = math.sqrt(n1 * n1 + n2 * n2 + n3 * n3)
        result[0] = n1 / d
        result[1] = n2 / d
        result[2] = n3 / d
        return result

    if branch_quadratic:
        # continue24: build quadratic-in-z coefficients
        fac1 = 1.0 / (wp[j][k].z - wp[j][k - 1].z)
        fac2 = 1.0 / (wp[j][k + 1].z - wp[j][k - 1].z)
        fac3 = 1.0 / (wp[j][k + 1].z - wp[j][k].z)
        fac4 = wp[j][k - 1].z + wp[j][k].z
        a2 = (a2 - a1) * fac1
        b2 = (b2 - b1) * fac1
        c2 = (c2 - c1) * fac1
        a3 = (a3 - a1) * fac2
        b3 = (b3 - b1) * fac2
        c3 = (c3 - c1) * fac2
        a3 = (a3 - a2) * fac3
        b3 = (b3 - b2) * fac3
        c3 = (c3 - c2) * fac3
        a2 = a2 - a3 * fac4
        b2 = b2 - b3 * fac4
        c2 = c2 - c3 * fac4
        a1 -= (a2 + a3 * wp[j][k - 1].z) * wp[j][k - 1].z
        b1 -= (b2 + b3 * wp[j][k - 1].z) * wp[j][k - 1].z
        c1 -= (c2 + c3 * wp[j][k - 1].z) * wp[j][k - 1].z
        ax = a1 + (a2 + a3 * P.z) * P.z
        by = b1 + (b2 + b3 * P.z) * P.z
        if bf2["type"] == CIRCLE:
            n2 = -2.0 * (P.x - a1 - (a2 + a3 * P.z) * P.z)
            n2 = abs(n2) if (P.x - ax < 0) else -abs(n2)
            n3 = -2.0 * (P.y - b1 - (b2 + b3 * P.z) * P.z)
            n3 = abs(n3) if (P.y - by < 0) else -abs(n3)
            n1 = -n2 * (a2 + 2.0 * a3 * P.z) - n3 * (b2 + 2 * b3 * P.z) + c2 + 2.0 * c3 * P.z
            if P.z == 0.0:
                n1 = abs(n1)
        else:
            n2 = abs(ax) if (P.x - ax < 0) else -abs(ax)
            n3 = abs(by) if (P.y - by < 0) else -abs(by)
            n1 = -(a2 + 2.0 * a3 * P.z) * P.x - (b2 + 2.0 * b3 * P.z) * P.y + (c2 + 2.0 * c3 * P.z)
        return _finish(n1, n2, n3)

    # continue27: general mixed-shape path
    f = [[0.0] * 6 for _ in range(4)]

    def _set_f(idx, bf, a, b, c):
        if bf["type"] != CIRCLE:
            f[idx][1] = 0.0
            f[idx][2] = a
            f[idx][3] = 0.0
            f[idx][4] = b
            f[idx][5] = c
        else:
            f[idx][1] = 1.0
            f[idx][2] = -2 * a
            f[idx][3] = 1.0
            f[idx][4] = -2 * b
            f[idx][5] = c - a * a - b * b

    _set_f(1, bf1, a1, b1, c1)
    _set_f(2, bf2, a2, b2, c2)
    _set_f(3, bf3, a3, b3, c3)

    dz21 = wp[0][k].z - wp[0][k - 1].z
    dz31 = wp[0][k + 1].z - wp[0][k - 1].z
    dz32 = wp[0][k + 1].z - wp[0][k].z
    z12 = wp[0][k - 1].z + wp[0][k].z
    fa = [0.0] * 6
    fb = [0.0] * 6
    fc = [0.0] * 6
    for ii in range(1, 6):
        fb[ii] = (f[2][ii] - f[1][ii]) / dz21
        fc[ii] = ((f[3][ii] - f[1][ii]) / dz31 - fb[ii]) / dz32
        fb[ii] += -fc[ii] * z12
        fa[ii] = f[1][ii] - (fb[ii] + fc[ii] * wp[0][k - 1].z) * wp[0][k - 1].z
    n2 = 2.0 * (fa[1] + (fb[1] + fc[1] * P.z) * P.z) * P.x + fa[2] + (fb[2] + fc[2] * P.z) * P.z
    n3 = 2.0 * (fa[3] + (fb[3] + fc[3] * P.z) * P.z) * P.y + fa[4] + (fb[4] + fc[4] * P.z) * P.z
    n1 = (-(fb[1] + 2.0 * fc[1] * P.z) * P.x * P.x - (fb[2] + 2.0 * fc[2] * P.z) * P.x
          - (fb[3] + 2.0 * fc[3] * P.z) * P.y * P.y - (fb[4] + 2.0 * fc[4] * P.z) * P.y
          + (fb[5] + 2.0 * fc[5] * P.z))
    ax = fa[1] + (fb[1] + fc[1] * P.z) * P.z
    if ax == 0.0:
        ax = -2.0
    ax = -2 * (fa[2] + (fb[2] + fc[2] * P.z * P.z)) / ax
    by = fa[3] + (fb[3] + fc[3] * P.z) * P.z
    if by == 0.0:
        by = -2
    by = -2 * (fa[4] + (fb[4] + fc[3] * P.z) * P.z) / by
    n2 = abs(n2) if (P.x - ax < 0) else -abs(n2)
    n3 = abs(n3) if (P.y - by < 0) else -abs(n3)
    return _finish(n1, n2, n3)


# ---------------------------------------------------------------------------
#  Compatibility-equation solvers
# ---------------------------------------------------------------------------
def compatibility_field(grid, base_pt, dtdN, dpdN, dzdN, theta, beta, delta, dz):
    """Port of ``CompatabilityEquationSolverForFieldPoint``."""
    PTN = [[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]]
    X = np.zeros(3)
    for j in range(4):
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for i in range(3):
            kk = PTN[j][i]
            a0, a1, a2, rhs = comp_equ(beta[kk], base_pt[kk], delta[kk],
                                       dzdN[kk], dz, theta[kk], dtdN[kk], dpdN[kk])
            A[i, 0] = a0
            A[i, 1] = a1
            A[i, 2] = a2
            b[i] = rhs
        sol = np.linalg.solve(A, b)
        X += sol
    return X / 4.0


def _body_funcv(x, a):
    return [
        a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2] - a[0][3],
        a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2] - a[1][3],
        a[2][0] * math.cos(x[2]) + a[2][1] * math.tan(x[1]) + a[2][2] * math.sin(x[2]),
    ]


def _body_jac(x, a):
    return [
        [a[0][0], a[0][1], a[0][2]],
        [a[1][0], a[1][1], a[1][2]],
        [0.0, a[2][1] / (math.cos(x[1]) ** 2),
         -a[2][0] * math.sin(x[2]) + a[2][2] * math.cos(x[2])],
    ]


def compatibility_body(grid, base_pt, dtdN, dpdN, dzdN, theta, beta, delta, dz,
                        P2: XYZPoint, unit_normal):
    """Port of ``CompatabilityEquationSolverForBodyPoint`` (Newton-Raphson)."""
    pt = [[0, 1], [0, 2], [1, 2]]
    X = np.zeros(3)
    a = grid._a
    a[2][0] = unit_normal[0]
    a[2][1] = unit_normal[1]
    a[2][2] = unit_normal[2]
    n = 2
    for j in range(n):
        x0 = [P2.p, P2.theta, P2.psi]
        for i in range(2):
            kk = pt[j][i]
            a0, a1, a2, rhs = comp_equ(beta[kk], base_pt[kk], delta[kk],
                                       dzdN[kk], dz, theta[kk], dtdN[kk], dpdN[kk])
            a[i][0] = a0
            a[i][1] = a1
            a[i][2] = a2
            a[i][3] = rhs
        a_copy = [row[:] for row in a]
        if _HAVE_SCIPY:
            sol = fsolve(_body_funcv, x0, args=(a_copy,),
                         fprime=_body_jac, xtol=1e-10, full_output=False)
        else:  # pragma: no cover
            sol = _newton_fallback(x0, a_copy)
        X += np.array(sol)
    return X / float(n)


def _newton_fallback(x0, a):  # pragma: no cover
    x = list(x0)
    for _ in range(200):
        f = _body_funcv(x, a)
        J = _body_jac(x, a)
        dx = np.linalg.solve(np.array(J), -np.array(f))
        x = [x[i] + dx[i] for i in range(3)]
        if max(abs(v) for v in f) < 1e-10:
            break
    return x


# ---------------------------------------------------------------------------
#  Field point
# ---------------------------------------------------------------------------
def calc_field_point(grid: "MOC3DGrid", i: int, k: int, dz: float) -> bool:
    """Port of ``CalcFieldPoint``."""
    P1 = grid._pt[i][k].clone()
    P2 = P1.clone()
    base = [P1.clone() for _ in range(4)]
    delta_i = [0.0] * 4

    g = P1.g
    t_total = P1.t * (1 + (g - 1) / 2 * P1.mach * P1.mach)
    p_total = P1.p * (1 + (g - 1) / 2 * P1.mach * P1.mach) ** (g / (g - 1))
    P2.z = P1.z + dz

    base[0].delta = 0.0
    if P1.x != 0.0 or P1.y != 0.0:
        base[0].delta = math.atan2(P1.x, P1.y)
    if base[0].delta < 0.0:
        base[0].delta += 2 * PI
    delta_i[0] = base[0].delta
    for ii in range(1, 4):
        base[ii].delta = base[ii - 1].delta + PI / 2
        if base[ii].delta >= 2 * PI:
            base[ii].delta -= 2 * PI
        delta_i[ii] = base[ii].delta

    p_err = theta_err = psi_err = 9.9
    n = 0
    dtdN = [0.0] * 4
    dpdN = [0.0] * 4
    dzdN = [0.0] * 4
    theta_avg = [0.0] * 4
    delta_avg = [0.0] * 4
    beta_avg = [0.0] * 4
    while ((abs(p_err) > 1e-5 or abs(theta_err) > 1e-5 or abs(psi_err) > 1e-5)
           and n < 50):
        n += 1
        theta12 = (P1.theta + P2.theta) / 2
        psi12 = (P1.psi + P2.psi) / 2
        P2.x = P1.x + math.tan(theta12) * dz / math.cos(psi12)
        P2.y = P1.y + math.tan(psi12) * dz
        if abs(P2.x) < 1e-5:
            P2.x = 0.0
        if abs(P2.y) < 1e-5:
            P2.y = 0.0

        for ib in range(4):
            beta = (_calc_mu(base[ib].mach) + _calc_mu(P2.mach)) / 2.0
            psi = (base[ib].psi + P2.psi) / 2.0
            theta = (base[ib].theta + P2.theta) / 2.0
            de = (base[ib].delta + delta_i[ib]) / 2
            base[ib].L = dz / (math.cos(beta) * math.cos(theta) * math.cos(psi)
                               - math.sin(beta) * (math.sin(theta) * math.cos(psi) * math.cos(de)
                                                   + math.sin(psi) * math.sin(de)))
            base[ib].x = P2.x - base[ib].L * (math.cos(beta) * math.sin(theta)
                                              + math.sin(beta) * math.cos(theta) * math.cos(de))
            base[ib].y = P2.y - base[ib].L * (math.cos(beta) * math.cos(theta) * math.sin(psi)
                                              - math.sin(beta) * (math.sin(theta) * math.sin(psi) * math.cos(de)
                                                                  - math.cos(psi) * math.sin(de)))
            base[ib].z = P1.z
            base_rad = math.sqrt(base[ib].x ** 2 + base[ib].y ** 2)
            if base_rad > grid._wall_pt[0][k].x:
                return False

            deriv = grid._calc_surface_point_properties(base[ib], i, k)
            dtdx = deriv[3]; dtdy = deriv[6]; dpdx = deriv[4]; dpdy = deriv[8]
            delta_i[ib] = calc_parametric_angle(beta, theta, psi, base[ib].delta)
            if abs(delta_i[ib]) <= 1e-8:
                delta_i[ib] = 0.0
            beta = (_calc_mu(base[ib].mach) + _calc_mu(P2.mach)) / 2.0
            psi = (base[ib].psi + P2.psi) / 2.0
            theta = (base[ib].theta + P2.theta) / 2.0
            de = (base[ib].delta + delta_i[ib]) / 2
            dxdN = -math.cos(theta) * math.sin(de)
            dydN = math.sin(theta) * math.sin(psi) * math.sin(de) + math.cos(psi) * math.cos(de)
            dzdN[ib] = math.sin(theta) * math.cos(psi) * math.sin(de) - math.sin(psi) * math.cos(de)
            dtdz = (dtdx * (base[ib].x - P2.x) + dtdy * (base[ib].y - P2.y)) / dz
            dpdz = (dpdx * (base[ib].x - P2.x) + dpdy * (base[ib].y - P2.y)) / dz
            dtdN[ib] = dtdx * dxdN + dtdy * dydN + dtdz * dzdN[ib]
            dpdN[ib] = dpdx * dxdN + dpdy * dydN + dpdz * dzdN[ib]
            beta_avg[ib] = beta
            theta_avg[ib] = theta
            delta_avg[ib] = de

        sol = compatibility_field(grid, base, dtdN, dpdN, dzdN,
                                  theta_avg, beta_avg, delta_avg, dz)
        new_p, new_theta, new_psi = sol[0], sol[1], sol[2]
        p_err = (new_p - P2.p) / P2.p
        theta_err = (new_theta - P2.theta) / P2.theta if abs(P2.theta) > 2e-2 else new_theta - P2.theta
        psi_err = (new_psi - P2.psi) / P2.psi if abs(P2.psi) > 2e-2 else new_psi - P2.psi
        P2.p = new_p
        P2.theta = new_theta
        P2.psi = new_psi
        P2.g = g
        P2.mol_wt = P1.mol_wt
        P2.t = t_total * (P2.p / p_total) ** ((g - 1) / g)
        P2.rho = (P2.p * 144) / (GASCON / P2.mol_wt * P2.t)
        P2.q = math.sqrt(2 * g / (g - 1) * GASCON / P2.mol_wt * GRAV * (t_total - P2.t))
        P2.mach = P2.q / math.sqrt(g * GASCON / P2.mol_wt * P2.t * GRAV)
        if P2.mach < 1.0:
            raise RuntimeError("Field point Mach < 1.0 in CalcFieldPoint")

    if abs(P2.theta) < 1e-5:
        P2.theta = 0.0
    if abs(P2.psi) < 1e-5:
        P2.psi = 0.0
    grid._pt[i][k + 1].copy_from(P2)
    return True


# ---------------------------------------------------------------------------
#  Body point
# ---------------------------------------------------------------------------
def calc_body_point(grid: "MOC3DGrid", i: int, k: int, dz: float) -> int:
    """Port of ``CalcBodyPoint``."""
    P1 = grid._pt[i][k].clone()
    P2 = P1.clone()
    base = [P1.clone() for _ in range(3)]
    delta_i = [0.0] * 3

    g = P1.g
    t_total = P1.t * (1 + (g - 1) / 2 * P1.mach * P1.mach)
    p_total = P1.p * (1 + (g - 1) / 2 * P1.mach * P1.mach) ** (g / (g - 1))
    P2.z += dz

    bfit2 = body_fit_X(grid, P2)
    un1 = calc_unit_normal_to_body_surface(grid, P1)
    un2 = calc_unit_normal_to_body_surface(grid, P2)

    N1, N2, N3 = un1[0], un1[1], un1[2]
    cos_delta = (-N1 * math.sin(P2.theta) * math.cos(P2.psi) + N2 * math.cos(P2.theta)
                 - N3 * math.sin(P2.theta) * math.sin(P2.psi))
    cos_delta = max(-1.0, min(1.0, cos_delta))
    base[0].delta = math.acos(cos_delta)
    if N3 < 0.0:
        base[0].delta = 2 * PI - base[0].delta
    base[1].delta = base[0].delta - PI / 3.0
    if base[1].delta < 0.0:
        base[1].delta += 2 * PI
    base[2].delta = base[0].delta + PI / 3.0
    if base[2].delta >= 2 * PI:
        base[2].delta -= 2 * PI
    for ib in range(3):
        delta_i[ib] = base[ib].delta

    p_err = theta_err = psi_err = 9.9
    n = 0
    dtdN = [0.0] * 3
    dpdN = [0.0] * 3
    dzdN = [0.0] * 3
    theta_avg = [0.0] * 3
    delta_avg = [0.0] * 3
    beta_avg = [0.0] * 3
    new_theta = P2.theta
    new_psi = P2.psi
    while ((abs(p_err) > 1e-5 or abs(theta_err) > 1e-5 or abs(psi_err) > 1e-5)
           and n < 50):
        n += 1
        theta12 = (P1.theta + P2.theta) / 2
        psi12 = (P1.psi + P2.psi) / 2
        N1 = (un1[0] + un2[0]) / 2
        N2 = (un1[1] + un2[1]) / 2
        N3 = (un1[2] + un2[2]) / 2
        CC = {
            "A": N3 * math.cos(theta12) * math.cos(psi12) - N1 * math.cos(theta12) * math.sin(psi12),
            "B": N1 * math.sin(theta12) - N2 * math.cos(theta12) * math.cos(psi12),
        }
        CC["C"] = (dz * (N3 * math.sin(theta12) - N2 * math.cos(theta12) * math.sin(psi12))
                   + CC["A"] * P1.x + CC["B"] * P1.y)
        new_xy = solve_for_body_point_position(CC, bfit2, P2.x, P2.y)
        P2.x, P2.y = new_xy
        if abs(P2.x) < 1e-5:
            P2.x = 0.0
        if abs(P2.y) < 1e-5:
            P2.y = 0.0

        new_jk = check_new_point_position(grid, P2, un2["j"], un2["k"])
        if new_jk[0] != un2["j"] and new_jk[1] != un2["k"]:
            bfit2 = body_fit_X(grid, P2)

        un2 = calc_unit_normal_to_body_surface(grid, P2)
        N1, N2, N3 = un2[0], un2[1], un2[2]
        cos_delta = (-N1 * math.sin(P2.theta) * math.cos(P2.psi) + N2 * math.cos(P2.theta)
                     - N3 * math.sin(P2.theta) * math.sin(P2.psi))
        cos_delta = max(-1.0, min(1.0, cos_delta))
        base[0].delta = math.acos(cos_delta)
        if N3 < 0.0:
            base[0].delta = 2 * PI - base[0].delta
        base[1].delta = base[0].delta - PI / 3.0
        if base[1].delta < 0.0:
            base[1].delta += 2 * PI
        base[2].delta = base[0].delta + PI / 3.0
        if base[2].delta >= 2 * PI:
            base[2].delta -= 2 * PI

        for ib in range(3):
            beta = (_calc_mu(base[ib].mach) + _calc_mu(P2.mach)) / 2.0
            psi = (base[ib].psi + P2.psi) / 2.0
            theta = (base[ib].theta + P2.theta) / 2.0
            de = (base[ib].delta + delta_i[ib]) / 2
            de -= PI
            if de < 0.0:
                de += 2 * PI
            base[ib].L = dz / (math.cos(beta) * math.cos(theta) * math.cos(psi)
                               - math.sin(beta) * (math.sin(theta) * math.cos(psi) * math.cos(de)
                                                   + math.sin(psi) * math.sin(de)))
            base[ib].x = P2.x - base[ib].L * (math.cos(beta) * math.sin(theta)
                                              + math.sin(beta) * math.cos(theta) * math.cos(de))
            base[ib].y = P2.y - base[ib].L * (math.cos(beta) * math.cos(theta) * math.sin(psi)
                                              - math.sin(beta) * (math.sin(theta) * math.sin(psi) * math.cos(de)
                                                                  - math.cos(psi) * math.sin(de)))
            base[ib].z = P1.z

            deriv = grid._calc_surface_point_properties(base[ib], i, k)
            dtdx = deriv[3]; dtdy = deriv[6]; dpdx = deriv[4]; dpdy = deriv[8]
            delta_i[ib] = calc_parametric_angle(beta, theta, psi, base[ib].delta)
            if abs(delta_i[ib]) <= 1e-8:
                delta_i[ib] = 0.0
            beta = (_calc_mu(base[ib].mach) + _calc_mu(P2.mach)) / 2.0
            psi = (base[ib].psi + P2.psi) / 2.0
            theta = (base[ib].theta + P2.theta) / 2.0
            de = (base[ib].delta + delta_i[ib]) / 2
            de -= PI
            if de < 0.0:
                de += 2 * PI
            dxdN = -math.cos(theta) * math.sin(de)
            dydN = math.sin(theta) * math.sin(psi) * math.sin(de) + math.cos(psi) * math.cos(de)
            dzdN[ib] = math.sin(theta) * math.cos(psi) * math.sin(de) - math.sin(psi) * math.cos(de)
            dtdz = (dtdx * (base[ib].x - P2.x) + dtdy * (base[ib].y - P2.y)) / dz
            dpdz = (dpdx * (base[ib].x - P2.x) + dpdy * (base[ib].y - P2.y)) / dz
            dtdN[ib] = dtdx * dxdN + dtdy * dydN + dtdz * dzdN[ib]
            dpdN[ib] = dpdx * dxdN + dpdy * dydN + dpdz * dzdN[ib]
            beta_avg[ib] = beta
            theta_avg[ib] = theta
            delta_avg[ib] = de

        sol = compatibility_body(grid, base, dtdN, dpdN, dzdN,
                                 theta_avg, beta_avg, delta_avg, dz, P2, un2)
        new_p, new_theta, new_psi = sol[0], sol[1], sol[2]
        p_err = (new_p - P2.p) / P2.p
        theta_err = (new_theta - P2.theta) / P2.theta if abs(P2.theta) > 2e-2 else new_theta - P2.theta
        psi_err = (new_psi - P2.psi) / P2.psi if abs(P2.psi) > 2e-2 else new_psi - P2.psi
        P2.p = new_p
        P2.theta = new_theta
        P2.psi = new_psi
        P2.g = P1.g
        P2.mol_wt = P1.mol_wt
        P2.t = t_total * (P2.p / p_total) ** ((g - 1) / g)
        P2.rho = (P2.p * 144) / (GASCON / P2.mol_wt * P2.t)
        P2.q = math.sqrt(2 * g / (g - 1) * GASCON / P2.mol_wt * GRAV * (t_total - P2.t))
        P2.mach = P2.q / math.sqrt(g * GASCON / P2.mol_wt * P2.t * GRAV)
        if P2.mach < 1.0:
            return FAIL_MACH

    if n == 50 and abs(p_err) > 1e-3 and abs(theta_err) > 1e-3 and abs(psi_err) > 1e-3:
        return FAIL

    if abs(P2.theta) < 1e-4:
        P2.theta = 0.0
        P2.x = P1.x
    if abs(P2.psi) < 1e-4:
        P2.psi = 0.0
        P2.y = P1.y

    grid._pt[i][k + 1].copy_from(P2)
    if grid._outfile:
        from .io_writers import _fmt
        grid._outfile.write(
            f"{k}\t{i}\t{_fmt(N1)}\t{_fmt(N2)}\t{_fmt(N3)}\n")
    return OK
