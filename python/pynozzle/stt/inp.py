"""Reader and writer for the ``.inp`` file format used by ``STT2001``.

The original file is whitespace-separated and has 19 lines (see
``CSTT2001Dlg::OnFileSave`` in ``STT2001Dlg.cpp``). The format groups
the five throat-constraint slots together so they can be read with a
simple ``ifile >> ...`` chain. We preserve the line and field order
exactly so ``.inp`` files round-trip with the Windows tool.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union


@dataclass
class ConstraintSlot:
    """One of the five throat/constraint slots in an STT2001 input file.

    ``rc``, ``yc``, ``zc`` describe a circular throat boundary in the YZ
    plane; ``alpha`` / ``omega`` are the start/end rotation angles (deg);
    ``nSL`` is the number of streamlines to find through the throat;
    ``x_start``/``x_end`` are the X-range over which a non-throat
    constraint is applied; ``surface`` is 0 for the inner surface, 1
    for outer (matches the C++ ``mC?_COMBO.GetCurSel()`` values);
    ``throat`` true means use as a throat boundary; ``constraint`` true
    means apply as an X-range trim.
    """
    rc: float = 0.0
    yc: float = 0.0
    zc: float = 0.0
    alpha: float = 0.0
    omega: float = 0.0
    nSL: int = 0
    x_start: float = 0.0
    x_end: float = 0.0
    surface: int = 0    # 0 = inner, 1 = outer
    throat: bool = False
    constraint: bool = False


@dataclass
class SymmetryDef:
    """Optional repeated-revolution symmetry settings (the 'SymDef' dialog)."""
    R_sim: float = 0.0
    Y_sim: float = 0.0
    Z_sim: float = 0.0
    n_rev: int = 1
    sl1_match: int = 0
    sl2_match: int = 0


@dataclass
class STTInput:
    """Parsed contents of an STT2001 ``.inp`` file.

    The fields are organised the same way the original GUI was: a set of
    file references on line 1, range/step inputs for the SL flow-field
    location, five constraint slots, and a few global parameters.
    """
    # file references (line 1)
    file_prefix: str = "case"
    sl_filename: str = "MOC_SL.plt"
    moc_grid_file: str = "MOC_Grid.plt"
    moc_summary_filename: str = "summary.out"
    friction_file: str = ""

    # SL-location sweeps (lines 2-5)
    RSL_start: float = 1.0
    RSL_end: float = 1.0
    RSL_step: float = 1.0
    XSL_start: float = 0.0
    XSL_end: float = 0.0
    XSL_step: float = 0.0
    YSL_start: float = 0.0
    YSL_end: float = 0.0
    YSL_step: float = 0.0
    ZSL_start: float = 0.0
    ZSL_end: float = 0.0
    ZSL_step: float = 0.0

    # constraint slots
    slots: list[ConstraintSlot] = field(
        default_factory=lambda: [ConstraintSlot() for _ in range(5)]
    )

    # globals (line 14)
    p_ambient: float = 0.0
    a_throat: float = 0.0
    isp_ideal: float = 100.0
    mass_flow: float = 0.0

    # symmetry block (line 15)
    sym: SymmetryDef = field(default_factory=SymmetryDef)

    # final options (line 19)
    max_length_check: bool = False
    max_length_c: float = 0.0
    grid_sf: int = 1
    x_status: float = 0.0
    contour_flag: bool = False


def _tok(text: str) -> list[str]:
    return [tok for tok in text.replace("\t", " ").split() if tok]


def read_inp(path: Union[str, Path]) -> STTInput:
    """Read an ``STT2001`` ``.inp`` file from disk.

    The legacy parser uses ``ifile >> ...`` which means whitespace is
    insignificant -- we treat the file as a long token list and assign
    in the same order the C++ does.
    """
    text = Path(path).read_text()
    # File-reference line is special: 5 tokens but they may contain
    # arbitrary characters apart from whitespace.  We parse it from the
    # *first physical line*; the rest is tokenised normally.
    raw_lines = text.splitlines()
    if len(raw_lines) < 2:
        raise ValueError(f"{path}: file appears truncated.")

    head_tokens = _tok(raw_lines[0])
    # The save routine writes 5 fields on line 1, separated by tabs.
    while len(head_tokens) < 5:
        head_tokens.append("")
    rest_tokens = _tok("\n".join(raw_lines[1:]))

    inp = STTInput()
    inp.file_prefix          = head_tokens[0]
    inp.sl_filename          = head_tokens[1]
    inp.moc_grid_file        = head_tokens[2]
    inp.moc_summary_filename = head_tokens[3]
    inp.friction_file        = head_tokens[4]

    it = iter(rest_tokens)
    def nf() -> float: return float(next(it))
    def ni() -> int:   return int(float(next(it)))

    # SL location sweep
    inp.RSL_start, inp.RSL_end, inp.RSL_step = nf(), nf(), nf()
    inp.XSL_start, inp.XSL_end, inp.XSL_step = nf(), nf(), nf()
    inp.YSL_start, inp.YSL_end, inp.YSL_step = nf(), nf(), nf()
    inp.ZSL_start, inp.ZSL_end, inp.ZSL_step = nf(), nf(), nf()

    # Five constraint slots - one field at a time across slots
    for k in range(5): inp.slots[k].rc      = nf()
    for k in range(5): inp.slots[k].yc      = nf()
    for k in range(5): inp.slots[k].zc      = nf()
    for k in range(5): inp.slots[k].alpha   = nf()
    for k in range(5): inp.slots[k].omega   = nf()
    for k in range(5): inp.slots[k].nSL     = ni()
    for k in range(5): inp.slots[k].x_start = nf()
    for k in range(5): inp.slots[k].x_end   = nf()

    # Globals
    inp.p_ambient = nf()
    inp.a_throat  = nf()
    inp.isp_ideal = nf()
    inp.mass_flow = nf()

    # Symmetry block
    inp.sym.R_sim     = nf()
    inp.sym.Y_sim     = nf()
    inp.sym.Z_sim     = nf()
    inp.sym.n_rev     = ni()
    inp.sym.sl1_match = ni()
    inp.sym.sl2_match = ni()

    for k in range(5): inp.slots[k].throat     = bool(ni())
    for k in range(5): inp.slots[k].constraint = bool(ni())
    for k in range(5): inp.slots[k].surface    = ni()

    inp.max_length_check = bool(ni())
    inp.max_length_c     = nf()
    inp.grid_sf          = ni()
    inp.x_status         = nf()
    try:
        inp.contour_flag = bool(ni())
    except StopIteration:
        inp.contour_flag = False

    return inp


def write_inp(inp: STTInput, path: Union[str, Path]) -> None:
    """Write an STT2001 input file in the legacy format."""
    buf = io.StringIO()
    def row(*xs): buf.write("\t".join(str(x) for x in xs) + "\n")

    row(inp.file_prefix, inp.sl_filename, inp.moc_grid_file,
        inp.moc_summary_filename, inp.friction_file)
    row(inp.RSL_start, inp.RSL_end, inp.RSL_step)
    row(inp.XSL_start, inp.XSL_end, inp.XSL_step)
    row(inp.YSL_start, inp.YSL_end, inp.YSL_step)
    row(inp.ZSL_start, inp.ZSL_end, inp.ZSL_step)
    row(*(s.rc      for s in inp.slots))
    row(*(s.yc      for s in inp.slots))
    row(*(s.zc      for s in inp.slots))
    row(*(s.alpha   for s in inp.slots))
    row(*(s.omega   for s in inp.slots))
    row(*(s.nSL     for s in inp.slots))
    row(*(s.x_start for s in inp.slots))
    row(*(s.x_end   for s in inp.slots))
    row(inp.p_ambient, inp.a_throat, inp.isp_ideal, inp.mass_flow)
    row(inp.sym.R_sim, inp.sym.Y_sim, inp.sym.Z_sim,
        inp.sym.n_rev, inp.sym.sl1_match, inp.sym.sl2_match)
    row(*(int(s.throat)     for s in inp.slots))
    row(*(int(s.constraint) for s in inp.slots))
    row(*(s.surface         for s in inp.slots))
    row(int(inp.max_length_check), inp.max_length_c,
        inp.grid_sf, inp.x_status, int(inp.contour_flag))

    Path(path).write_text(buf.getvalue())
