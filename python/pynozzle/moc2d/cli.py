"""Command-line interface for :mod:`pynozzle.moc2d`.

Replaces the original MFC GUI of ``MOC_Grid_BDE``. Usage::

    pynozzle-moc2d  case.inp  [--output-dir DIR]  [--full]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .inp import read_inp, MOCInput
from .solver import MOCGridCalc
from .io_writers import write_all


def _build_calc(inp: MOCInput, full_output: bool) -> MOCGridCalc:
    """Translate an :class:`MOCInput` into a configured :class:`MOCGridCalc`."""
    calc = MOCGridCalc()
    ok = calc.set_initial_properties(
        pres=inp.pres_i,
        temp=inp.temp_i,
        mol_wt=inp.mol_wt_i,
        gamma=inp.gamma_i,
        p_amb=inp.p_amb,
        n=inp.n_c,
        rwt_u=inp.rwt_u,
        rwt_d=inp.rwt_d,
        d_t_limit_deg=inp.d_t_limit,
        n_rrc_above_bd=inp.n_rrc_above_bd,
        n_sl_i=inp.n_sl_i,
        n_sl_j=inp.n_sl_j,
        vel=inp.vel,
        throat_flag=int(inp.throat),
        isp_ideal=inp.isp_ideal,
    )
    if not ok:
        raise SystemExit("Could not set initial properties (check positive p/T/MW/gamma).")
    v1, v2 = inp.design_values()
    calc.set_solution_parameters(
        geom=inp.nozzle_geom(),
        nozzle_type=inp.nozzle_type(),
        design_param=inp.design_param(),
        value1=v1,
        value2=v2,
        theta_bi_deg=inp.theta_bi,
    )
    calc.set_print_mode(1 if full_output else 0)
    return calc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pynozzle-moc2d",
        description=(
            "2D / axisymmetric Method-of-Characteristics nozzle designer. "
            "Reads .inp files from the original MOC_Grid_BDE tool and writes "
            "the same output files (summary.out, MOC_Grid.plt, MOC_SL.plt, "
            "center.out, rao.dat)."
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a .inp file in the MOC_Grid_BDE format.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to write output files into (default: current dir).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Write only summary.out (matches the print_mode=0 behavior of "
            "the original GUI). Default writes the full grid as well."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    inp = read_inp(args.input)
    full_output = not args.summary_only
    calc = _build_calc(inp, full_output)
    result = calc.run()
    if not result.success:
        print("error:", result.error_message or "MOC solver failed", file=sys.stderr)
        return 1
    write_all(result, args.output_dir, full_output=full_output)
    print(f"Wrote results to {args.output_dir.resolve()}")
    print(f"  ThetaB(deg)     = {result.theta_b_ans * 180.0/3.141592653589793:.5g}")
    print(f"  last RRC        = {result.last_rrc}")
    print(f"  exit Mach (wall)= {result.grid.mach[0, result.last_rrc]:.5g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
