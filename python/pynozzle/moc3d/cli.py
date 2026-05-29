"""Command-line interface for :mod:`pynozzle.moc3d`.

Replaces the original MFC GUI of ``3D_MOC``. Usage::

    pynozzle-moc3d  case.geo  [options]  -o output_dir

The default flow parameters mirror the GUI defaults
(p0=1000, T0=530, M0=1.1, MW=28.96, gamma=1.4, theta=psi=0,
nDiv=36, zStep=10, surface fit = "All Point Spline").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .geo import GeoConfig, read_geo
from .solver import MOC3DGrid


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pynozzle-moc3d",
        description=(
            "3D Method-of-Characteristics flow-field solver. Reads a "
            ".geo wall-contour file (as written by MOC_Grid_BDE) and "
            "writes Tecplot-format full mesh, streamline, wall and "
            "axial-station plot files."
        ),
    )
    p.add_argument("geo", type=Path, help="Path to the .geo wall-contour file.")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("."),
                   help="Output directory (default: current directory).")
    p.add_argument("--p0", type=float, default=1000.0, help="Chamber pressure (psia)")
    p.add_argument("--t0", type=float, default=530.0, help="Chamber temperature (R)")
    p.add_argument("--mach0", type=float, default=1.1, help="Initial Mach number")
    p.add_argument("--mol-wt", type=float, default=28.96, help="Molecular weight")
    p.add_argument("--gamma", type=float, default=1.4, help="Ratio of specific heats")
    p.add_argument("--theta0", type=float, default=0.0, help="Initial flow angle theta (deg)")
    p.add_argument("--psi0", type=float, default=0.0, help="Initial flow angle psi (deg)")
    p.add_argument("--n-div", type=int, default=36, help="Angular divisions of the wall")
    p.add_argument("--x-step", type=int, default=1, help="X output stride")
    p.add_argument("--y-step", type=int, default=1, help="Y output stride")
    p.add_argument("--z-step", type=int, default=10, help="Z output stride")
    p.add_argument("--step-step", type=int, default=999, help="Intermediate-output stride")
    p.add_argument("--surface-fit", choices=("All Point Spline", "9 Point Spline"),
                   default="All Point Spline", help="Surface-fit method")
    args = p.parse_args(list(argv) if argv is not None else None)

    if not args.geo.is_file():
        print(f"error: .geo file not found: {args.geo}", file=sys.stderr)
        return 2

    cfg = GeoConfig(
        p0=args.p0, t0=args.t0, mach0=args.mach0,
        mol_wt0=args.mol_wt, gamma0=args.gamma,
        theta0=args.theta0, psi0=args.psi0,
        n_div=args.n_div,
        x_output_step=args.x_step, y_output_step=args.y_step,
        z_output_step=args.z_step, step_step=args.step_step,
        surface_fit=args.surface_fit,
    )
    read_geo(args.geo, cfg)
    grid = MOC3DGrid(cfg, args.output_dir)
    grid.set_initial_properties()
    result = grid.calc_nozzle()
    if not result.success:
        print("error:", result.error_message or "solver failed", file=sys.stderr)
        return 1
    print(f"Wrote results to {args.output_dir.resolve()}")
    print(f"  Total field points : {result.n_pts}")
    print(f"  Axial planes       : {result.n_z}")
    print(f"  Angular divisions  : {result.n_div}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
