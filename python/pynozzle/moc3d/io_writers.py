"""Output writers for the 3D MOC solver.

Reproduces the Tecplot files emitted by the original ``OutputContour``,
``OutputStreamlines``, ``OutputAxialStations`` and ``OutputBoundary``
routines. The C++ writes through plain ``ofile << x`` (default ostream
formatting, ~6 significant figures) -- we use a small helper to match.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from .point import DEG_PER_RAD

if TYPE_CHECKING:
    from .solver import MOC3DGrid


def _fmt(x: float) -> str:
    """C++ default ostream output (up to 6 significant digits)."""
    if x != x:
        return "0"
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e6 or ax < 1e-4:
        s = f"{x:.5e}"
        mant, exp = s.split("e")
        mant = mant.rstrip("0").rstrip(".")
        e = int(exp)
        sign = "-" if e < 0 else "+"
        # MSVC default: 3-digit exponent (e.g. ``8.74719e-018``)
        return f"{mant}e{sign}{abs(e):03d}"
    return f"{x:.6g}"


def _row(pt, i: int, k: int, body_flag: bool) -> str:
    ratio = 1 + (pt.g - 1.0) / 2.0 * pt.mach * pt.mach
    tt = pt.t * ratio
    pt_total = pt.p * pow(ratio, pt.g / (pt.g - 1))
    rad = math.sqrt(pt.x * pt.x + pt.y * pt.y)
    return (f"{_fmt(pt.x)}\t{_fmt(pt.y)}\t{_fmt(pt.z)}\t{_fmt(rad)}\t"
            f"{_fmt(pt.p)}\t{_fmt(pt_total)}\t{_fmt(pt.t)}\t{_fmt(tt)}\t"
            f"{_fmt(pt.rho)}\t{_fmt(pt.mach)}\t{_fmt(pt.q)}\t"
            f"{_fmt(pt.theta * DEG_PER_RAD)}\t{_fmt(pt.psi * DEG_PER_RAD)}\t"
            f"{i}\t{k}\t{int(body_flag)}\n")


def output_contour(grid: "MOC3DGrid", i_start: int, i_end: int,
                   k_start: int, k_end: int) -> None:
    """Port of ``OutputContour``: writes ``full_mesh.plt`` then chains
    streamlines, axial stations and the boundary."""
    xs = grid._x_step
    zs = grid._z_step
    ni = (i_end - i_start) // xs + 1
    nk = (k_end - k_start) // zs + 1

    path = grid.out_dir / "full_mesh.plt"
    with open(path, "w") as f:
        f.write('VARIABLES = "X(in)","Y(in)","Z(in)","R(in)","P(psia)",'
                '"PT(psia)","T(R)","TT(R)","RHO(lbm/ft3)","Mach",'
                '"Velocity(ft/s)","Theta(deg)","Psi(deg)","I","K","BodyFlag"\n')
        f.write('TITLE = "Volume Mesh Plot"\n')
        f.write('text  x=5  y=93  t="Characteristic Mesh Plot" \n')
        f.write(f'zone t="All Data" I = {ni} J = 1 K ={nk}\n')
        for k in range(k_start, k_end + 1, zs):
            for i in range(i_start, i_end + 1, xs):
                f.write(_row(grid._pt[i][k], i, k, grid._body_point_flag[i]))

    output_streamlines(grid, i_start, i_end, k_start, k_end)
    output_axial_stations(grid, i_start, i_end, k_start, k_end)
    output_boundary(grid, i_start, i_end, k_start, k_end)


def output_axial_stations(grid: "MOC3DGrid", i_start, i_end, k_start, k_end):
    xs = grid._x_step
    zs = grid._z_step
    path = grid.out_dir / "axialStations.plt"
    with open(path, "w") as f:
        f.write('VARIABLES = "X(in)","Y(in)","Z(in)","R(in)","P(psia)",'
                '"PT(psia)","T(R)","TT(R)","RHO(lbm/ft3)","Mach",'
                '"Velocity(ft/s)","Theta(deg)","Psi(deg)","I","K","BodyFlag"\n')
        f.write('TITLE = "Volume Mesh Plot"\n')
        f.write('text  x=5  y=93  t="Characteristic Mesh Plot" \n')
        k = k_start
        last_k_written = None
        while k <= k_end:
            f.write(f'zone t="Z = {k}" I = 1 J = 1 K = {(i_end-i_start)//xs+1}\n')
            for i in range(i_start, i_end + 1, xs):
                f.write(_row(grid._pt[i][k], i, k, grid._body_point_flag[i]))
            last_k_written = k
            k += zs
        if last_k_written is not None and last_k_written != k_end:
            k = k_end
            f.write(f'zone t="Z = {k}" I = 1 J = 1 K = {(i_end-i_start)//xs+1}\n')
            for i in range(i_start, i_end + 1, xs):
                f.write(_row(grid._pt[i][k], i, k, grid._body_point_flag[i]))


def output_streamlines(grid: "MOC3DGrid", i_start, i_end, k_start, k_end):
    xs = grid._x_step
    zs = grid._z_step
    path = grid.out_dir / "Streamlines.plt"
    with open(path, "w") as f:
        f.write('VARIABLES = "X(in)","Y(in)","Z(in)","R(in)","P(psia)",'
                '"PT(psia)","T(R)","TT(R)","RHO(lbm/ft3)","Mach",'
                '"Velocity(ft/s)","Theta(deg)","Psi(deg)","I","K","BodyFlag"\n')
        f.write('TITLE = "Streamline Plot"\n')
        f.write('text  x=5  y=93  t="Streamline Plot" \n')
        for i in range(i_start, i_end + 1, xs):
            f.write(f'zone t="I = {i}" I ={(k_end-k_start)//zs+1} J = 1 K = 1\n')
            last_k_written = None
            for k in range(k_start, k_end + 1, zs):
                f.write(_row(grid._pt[i][k], i, k, grid._body_point_flag[i]))
                last_k_written = k
            if last_k_written is not None and last_k_written != k_end:
                k = k_end
                f.write(_row(grid._pt[i][k], i, k, grid._body_point_flag[i]))

    # SL.inp -> streamtube.plt (if SL.inp present in CWD or input dir)
    sl_inp = None
    for candidate in (grid.out_dir / "SL.inp", Path.cwd() / "SL.inp"):
        if candidate.is_file():
            sl_inp = candidate
            break
    if sl_inp is None:
        return
    toks = sl_inp.read_text().split()
    if not toks:
        return
    n = int(float(toks[0]))
    pti = [int(float(t)) for t in toks[1:1 + n]]
    path = grid.out_dir / "streamtube.plt"
    with open(path, "w") as f:
        f.write('VARIABLES = "X(in)","Y(in)","Z(in)","R(in)","P(psia)",'
                '"PT(psia)","T(R)","TT(R)","RHO(lbm/ft3)","Mach",'
                '"Velocity(ft/s)","Theta(deg)","Psi(deg)","I","K","BodyFlag"\n')
        f.write('TITLE = "Volume Mesh Plot"\n')
        f.write('text  x=5  y=93  t="Characteristic Mesh Plot" \n')
        f.write(f'zone t="All data" I = {(k_end-k_start)//zs+1}'
                f' J = 1 K = {n+1}\n')
        for j in range(n):
            ii = pti[j]
            for k in range(k_start, k_end + 1, zs):
                f.write(_row(grid._pt[ii][k], ii, k, grid._body_point_flag[ii]))
        ii = pti[0]
        for k in range(k_start, k_end + 1, zs):
            f.write(_row(grid._pt[ii][k], ii, k, grid._body_point_flag[ii]))


def output_boundary(grid: "MOC3DGrid", i_start, i_end, k_start, k_end):
    zs = grid._z_step
    n_pts = sum(1 for i in range(i_start, i_end + 1) if grid._body_point_flag[i])
    path = grid.out_dir / "Wall.plt"
    with open(path, "w") as f:
        f.write('VARIABLES = "X(in)","Y(in)","Z(in)","R(in)","P(psia)",'
                '"PT(psia)","T(R)","TT(R)","RHO(lbm/ft3)","Mach",'
                '"Velocity(ft/s)","Theta(deg)","Psi(deg)","I","K","BodyFlag"\n')
        f.write('TITLE = "Volume Mesh Plot"\n')
        f.write('text  x=5  y=93  t="Characteristic Mesh Plot" \n')
        f.write(f'zone t="All data" I = {(k_end-k_start)//zs+2}'
                f' J = 1 K = {n_pts+1}\n')

        for i in range(i_start, i_end + 1):
            for k in range(k_start, k_end + 1, zs):
                if grid._body_point_flag[i]:
                    f.write(_row(grid._pt[i][k], i, k, True))
            if grid._body_point_flag[i]:
                f.write(_row(grid._pt[i][k_end], i, k_end, True))

        for i in range(i_start, i_end + 1):
            for k in range(k_start, k_end + 1, zs):
                if grid._body_point_flag[i]:
                    f.write(_row(grid._pt[i][k], i, k, True))
            if grid._body_point_flag[i]:
                f.write(_row(grid._pt[i][k_end], i, k_end, True))
            if grid._body_point_flag[i]:
                break
