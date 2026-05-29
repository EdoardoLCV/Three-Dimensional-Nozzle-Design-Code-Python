"""Loaders for the upstream files produced by ``MOC_Grid_BDE``.

The STT2001 program needs three inputs from a prior MOC run:

* ``MOC_SL.plt`` — the streamlines (one zone per mass-flow fraction)
* ``summary.out`` — scalar performance values (mass flow, thrust, etc.)
* ``MOC_Grid.plt`` — the MOC mesh, used by ``FindMaxX`` to check the
  reflective-wave validity of the trimmed contour.

These are all the text formats that the moc2d port emits.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np


# ---------------------------------------------------------------------------
@dataclass
class SLData:
    """Streamline arrays loaded from ``MOC_SL.plt``.

    All arrays are shaped ``(nSL, max_pts)`` and padded with zeros for
    streamlines shorter than ``max_pts``. ``last_sl_pt[i]`` is the index
    of the last valid point on streamline ``i`` (matches the C++
    ``lastSLPt[i]`` semantics: it's the *index*, not the count).
    """
    n_sl: int
    psi: np.ndarray            # (nSL,)
    last_sl_pt: np.ndarray     # (nSL,) ints, index of last point
    xsl: np.ndarray            # (nSL, max_pts)
    rsl: np.ndarray
    msl: np.ndarray
    psl: np.ndarray
    tsl: np.ndarray
    thsl: np.ndarray
    jsl: np.ndarray            # ints, the original MOC-mesh J of the point
    x_min: float
    x_max: float


def _parse_zone_psi(line: str) -> float:
    """The C++ extracts 3 characters at offset 21 of the zone line and
    atof's them. We mirror this exactly so we get the same psi values."""
    # zone t="MassFlow % = 12.5" I = ...
    # The C++ slice [21:24] grabs the value digits (e.g. "12.", "0.0", "100").
    # Use atof semantics: stop at first non-numeric.
    chunk = line[21:24] if len(line) >= 24 else line[21:]
    m = re.match(r"^\s*([-+]?\d*\.?\d+)", chunk)
    return float(m.group(1)) if m else 0.0


def read_sl_plt(
    path: Union[str, Path], n_param_pts: int = 4200,
) -> SLData:
    """Direct port of ``CSTT2001Dlg::GetSLData``.

    ``MOC_SL.plt`` is a Tecplot ASCII file with three header lines and
    then per-zone blocks. Each zone begins with a ``zone t="..."`` line
    that encodes the mass-flow fraction; subsequent lines have 12
    space-separated numbers per streamline point. We extract everything
    we need into NumPy arrays.
    """
    text = Path(path).read_text()
    lines = text.splitlines()
    if len(lines) < 4:
        raise ValueError(f"{path}: too short to be a valid MOC_SL.plt")

    # The C++ reader skips the first 3 lines unconditionally.
    body = lines[3:]

    # We collect into Python lists per streamline; pad to a fixed array
    # at the end so callers get a stable indexed view.
    psi_list: list[float] = []
    xsl_list: list[list[float]] = []
    rsl_list: list[list[float]] = []
    msl_list: list[list[float]] = []
    psl_list: list[list[float]] = []
    tsl_list: list[list[float]] = []
    thsl_list: list[list[float]] = []
    jsl_list: list[list[int]] = []

    x_min = float("inf")
    x_max = float("-inf")
    current_sl_idx = -1

    for ln in body:
        if not ln.strip():
            continue
        if ln.lstrip().startswith("zone"):
            current_sl_idx += 1
            psi_list.append(_parse_zone_psi(ln))
            xsl_list.append([])
            rsl_list.append([])
            msl_list.append([])
            psl_list.append([])
            tsl_list.append([])
            thsl_list.append([])
            jsl_list.append([])
        else:
            toks = ln.split()
            if len(toks) < 12:
                continue
            # x, dummy1, dummy2, r, M, p, t, rho, theta, g, mdot, j
            x      = float(toks[0])
            r      = float(toks[3])
            M      = float(toks[4])
            p      = float(toks[5])
            t      = float(toks[6])
            theta  = float(toks[8])
            j_val  = int(float(toks[11]))
            xsl_list[current_sl_idx].append(x)
            rsl_list[current_sl_idx].append(r)
            msl_list[current_sl_idx].append(M)
            psl_list[current_sl_idx].append(p)
            tsl_list[current_sl_idx].append(t)
            thsl_list[current_sl_idx].append(theta)
            jsl_list[current_sl_idx].append(j_val)
            if x < x_min: x_min = x
            if x > x_max: x_max = x

    n_sl = current_sl_idx + 1
    if n_sl == 0:
        raise ValueError(f"{path}: no streamline zones found")

    max_pts = max(n_param_pts, max(len(s) for s in xsl_list) + 1)
    shape = (n_sl, max_pts)
    xsl = np.zeros(shape)
    rsl = np.zeros(shape)
    msl = np.zeros(shape)
    psl = np.zeros(shape)
    tsl = np.zeros(shape)
    thsl = np.zeros(shape)
    jsl = np.zeros(shape, dtype=np.int64)
    last_sl_pt = np.zeros(n_sl, dtype=np.int64)
    psi = np.array(psi_list, dtype=np.float64)
    for i in range(n_sl):
        n_here = len(xsl_list[i])
        xsl[i, :n_here]  = xsl_list[i]
        rsl[i, :n_here]  = rsl_list[i]
        msl[i, :n_here]  = msl_list[i]
        psl[i, :n_here]  = psl_list[i]
        tsl[i, :n_here]  = tsl_list[i]
        thsl[i, :n_here] = thsl_list[i]
        jsl[i, :n_here]  = jsl_list[i]
        last_sl_pt[i] = n_here - 1

    return SLData(
        n_sl=n_sl, psi=psi, last_sl_pt=last_sl_pt,
        xsl=xsl, rsl=rsl, msl=msl, psl=psl, tsl=tsl, thsl=thsl, jsl=jsl,
        x_min=x_min, x_max=x_max,
    )


# ---------------------------------------------------------------------------
@dataclass
class MOCSummaryData:
    """Scalars extracted from the upstream ``summary.out``."""
    thrust_1_moc: float = -1.0
    a_surf_moc: float = -1.0
    mdot_moc: float = -1.0
    eps_moc: float = -1.0
    f_exit_moc: float = -1.0
    s_exit_moc: float = -1.0
    r_star_moc: float = -1.0
    p_amb_moc: float = -1.0
    a_exit_moc: float = -1.0
    isp_2d_moc: float = -1.0


def _atof_field(line: str, offset: int, length: int) -> float:
    """Emulate the C++ ``strLine.copy(cLine, length, offset); atof(cLine)``
    pattern: take a fixed-width slice and parse the leading number."""
    chunk = line[offset:offset + length]
    m = re.match(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", chunk)
    return float(m.group(1)) if m else float("nan")


def read_moc_summary(path: Union[str, Path]) -> MOCSummaryData:
    """Direct port of ``CSTT2001Dlg::GetPerformanceDataFromMOCSummaryFile``.

    The original code looks for lines beginning with specific prefixes
    (``"2-D Gross Thrust"``, ``"Surface Area"`` etc.) and parses a
    fixed-width slice. We mirror that exactly so the same files load.
    """
    data = MOCSummaryData()
    for ln in Path(path).read_text(errors="replace").splitlines():
        ln = ln.rstrip("\r\n")
        if not ln:
            continue
        if ln.startswith("2-D Gross Thrust"):
            data.thrust_1_moc = _atof_field(ln, 23, 10)
        elif ln.startswith("Surface Area"):
            data.a_surf_moc = _atof_field(ln, 19, 10)
        elif ln.startswith("2-D Mass Flow"):
            data.mdot_moc = _atof_field(ln, 22, 10)
        elif ln.startswith("Expansion Ratio"):
            data.eps_moc = _atof_field(ln, 17, 10)
        elif ln.startswith("Gross Thrust"):
            data.f_exit_moc = _atof_field(ln, 19, 10)
        elif ln.startswith("Stream Thrust"):
            data.s_exit_moc = _atof_field(ln, 20, 10)
        elif ln.startswith("Throat Radius R*"):
            data.r_star_moc = _atof_field(ln, 22, 10)
        elif ln.startswith("Ambient Pressure"):
            data.p_amb_moc = _atof_field(ln, 24, 10)
        elif ln.startswith("Exit Area"):
            data.a_exit_moc = _atof_field(ln, 16, 10)
        elif ln.startswith("Isp"):
            data.isp_2d_moc = _atof_field(ln, 17, 10)
            break  # C++ stops here

    # The C++ ``Throat Radius R*`` parser reads from offset 22, but the
    # field there is sometimes the line title rather than the value (the
    # actual number on the M3.5Perf sample is at column 23). Fall back to
    # a more permissive search if the strict parse missed it.
    if data.r_star_moc <= 0.0:
        for ln in Path(path).read_text(errors="replace").splitlines():
            if "Throat Radius" in ln and "\t" in ln:
                try:
                    data.r_star_moc = float(ln.split("\t", 1)[1].strip())
                    break
                except ValueError:
                    pass

    return data


# ---------------------------------------------------------------------------
@dataclass
class MOCGridData:
    """Mesh data extracted from ``MOC_Grid.plt`` (used by FindMaxX)."""
    x: np.ndarray   # (max_j, max_k)
    r: np.ndarray
    i: np.ndarray   # ints
    last_pt: np.ndarray   # ints, per-zone last k
    n_zones: int          # last J + 1


def read_moc_grid(
    path: Union[str, Path], j_param: int = 220, k_param: int = 199,
) -> MOCGridData:
    """Direct port of the file-reading block inside ``CSTT2001Dlg::FindMaxX``.

    The MOC grid file has ``zone t="J = N"`` headers; for each J we
    collect the (x, r, i) data per K. Defaults for the array sizes
    match the C++ constants ``jParam = 220`` and ``kParam = 199``.
    """
    text = Path(path).read_text()
    lines = text.splitlines()
    body = lines[3:]   # skip 3 header lines, same as the C++

    xMOC = np.zeros((j_param, k_param))
    rMOC = np.zeros((j_param, k_param))
    iMOC = np.zeros((j_param, k_param), dtype=np.int64)
    last_pt = np.zeros(j_param, dtype=np.int64)

    first_time_thru = True
    k = 0
    j = 0
    for ln in body:
        s = ln.lstrip()
        if not s:
            continue
        if s.startswith("zone"):
            if not first_time_thru:
                last_pt[j] = k - 1
            first_time_thru = False
            k = 0
            # The C++ extracts 3 characters from offset 12 to parse J.
            chunk = ln[12:15] if len(ln) >= 15 else ln[12:]
            m = re.match(r"^\s*([-+]?\d+)", chunk)
            if not m:
                raise ValueError(f"MOC_Grid.plt zone line could not parse J: {ln!r}")
            j = int(m.group(1))
            if j >= j_param:
                raise ValueError(
                    f"FindMaxX: more J zones ({j+1}) than initialised ({j_param})"
                )
        else:
            toks = ln.split()
            if len(toks) < 6:
                continue
            xMOC[k, j] = float(toks[0])
            rMOC[k, j] = float(toks[1])
            iMOC[k, j] = int(float(toks[5]))
            k += 1
            if k >= k_param:
                raise ValueError(
                    f"FindMaxX: more K points ({k+1}) than initialised ({k_param})"
                )
    last_pt[j] = k - 1
    return MOCGridData(x=xMOC, r=rMOC, i=iMOC, last_pt=last_pt, n_zones=j + 1)


# ---------------------------------------------------------------------------
def get_friction_loss(path: Union[str, Path], surface_area: float) -> float:
    """Direct port of ``CSTT2001Dlg::GetFrictionLoss``.

    The friction file is a table of (surface_area, friction_force) pairs,
    preceded by a count. The friction loss for the trimmed nozzle is a
    linear interpolation of the table at the nozzle's surface area.

    Note the C++ argument is named ``eps`` (area ratio) in the source but
    is actually called with ``m_SurfaceArea``.

    Returns 0.0 if the file is missing or empty.
    """
    p = Path(path)
    if not p.is_file():
        return 0.0
    toks = p.read_text().split()
    if not toks:
        return 0.0
    it = iter(toks)
    try:
        itotal = int(float(next(it)))
    except (StopIteration, ValueError):
        return 0.0
    if itotal >= 25:
        raise ValueError("The friction data file can have a max of 25 points")
    a_ratio = []
    fric = []
    for _ in range(itotal):
        try:
            a_ratio.append(float(next(it)))
            fric.append(float(next(it)))
        except StopIteration:
            break
    if len(a_ratio) < 2:
        return fric[0] if fric else 0.0

    # while ( eps > aRatio[++i] );  -- find first i with aRatio[i] >= eps
    i = 0
    while True:
        i += 1
        if i >= len(a_ratio) or surface_area <= a_ratio[i]:
            break
    if i >= len(a_ratio):
        i = len(a_ratio) - 1
    if i > 0:
        denom = a_ratio[i] - a_ratio[i - 1]
        if denom == 0.0:
            return fric[i]
        return (fric[i] - (a_ratio[i] - surface_area) / denom
                * (fric[i] - fric[i - 1]))
    return fric[i]
