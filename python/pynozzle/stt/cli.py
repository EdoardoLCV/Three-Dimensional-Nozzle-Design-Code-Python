"""Command-line interface for :mod:`pynozzle.stt`.

Replaces the original MFC GUI of ``STT2001``. Usage::

    pynozzle-stt  case.inp  [--input-dir DIR]  [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .inp import read_inp
from .solver import STTSolver


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pynozzle-stt",
        description=(
            "Streamline tracing tool. Reads .inp files from the original "
            "STT2001 tool and writes the same output files. Expects the "
            "upstream MOC_Grid_BDE outputs (MOC_SL.plt, MOC_Grid.plt, "
            "summary.out) to be present in --input-dir."
        ),
    )
    p.add_argument("input", type=Path,
                   help="Path to a .inp file in the STT2001 format.")
    p.add_argument("-i", "--input-dir", type=Path, default=None,
                   help="Where to look for MOC_SL.plt etc. (default: dir of input).")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("."),
                   help="Directory to write outputs into (default: current dir).")
    args = p.parse_args(list(argv) if argv is not None else None)

    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    input_dir = args.input_dir or args.input.parent
    inp = read_inp(args.input)
    solver = STTSolver(input_dir, inp)
    result = solver.run(args.output_dir)
    if not result.success:
        print("error:", result.error_message or "STT solver failed",
              file=sys.stderr)
        return 1
    print(f"Wrote results to {args.output_dir.resolve()}")
    print(f"  Surface area (in2) = {result.surface_area:.4g}")
    print(f"  Projected area (in2) = {result.projected_area:.4g}")
    print(f"  Pressure force (lbf) = {result.pressure_force:.4g}")
    print(f"  Throat thrust (lbf)  = {result.throat_thrust:.4g}")
    print(f"  Isp calc (lbf-s/lbm) = {result.isp_calc:.4g}")
    print(f"  Cfg                  = {result.cxx:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
