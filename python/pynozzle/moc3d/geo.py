"""``.geo`` wall-contour reader and run configuration for ``3D_MOC``.

The ``.geo`` file (written by the upstream tools) is a simple table:

    <nZ>
    X   R   Xo   Y0
    <z0> <r0> <x0> <y0>
    ... nZ rows ...

Each row defines a circle of radius ``r0`` centred at ``(x0, y0)`` in the
cross-section at axial station ``z0``. Stacking the circles along the
axis gives the nozzle wall (a body of revolution when ``x0 = y0 = 0``).

The remaining run parameters (chamber state, Mach, number of angular
divisions, etc.) come from the GUI in the original; here they live in
:class:`GeoConfig` and default to the values used for the sample cases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np


@dataclass
class GeoConfig:
    """Configuration for a 3D_MOC run.

    The wall geometry (``z``, ``r``, ``xc``, ``yc`` arrays) is read from a
    ``.geo`` file; the flow/initialisation parameters mirror the fields
    of the original GUI dialog (``CMy3D_MOCDlg``).
    """
    # Wall contour (filled by :func:`read_geo`)
    z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    r: np.ndarray = field(default_factory=lambda: np.zeros(0))
    xc: np.ndarray = field(default_factory=lambda: np.zeros(0))
    yc: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Chamber / initial-plane flow state
    p0: float = 1000.0       # chamber pressure (psia)
    t0: float = 530.0        # chamber temperature (R)
    mach0: float = 1.1       # initial-plane Mach number
    mol_wt0: float = 28.96   # molecular weight
    gamma0: float = 1.4      # ratio of specific heats
    theta0: float = 0.0      # initial flow angle theta (deg)
    psi0: float = 0.0        # initial flow angle psi (deg)

    # Discretisation / output controls
    n_div: int = 36          # number of angular divisions of the wall
    x_output_step: int = 1
    y_output_step: int = 1
    z_output_step: int = 10
    step_step: int = 999     # how often to dump intermediate contours
    surface_fit: str = "All Point Spline"  # or "9 Point Spline"

    @property
    def n_z(self) -> int:
        return int(self.z.size)


def read_geo(path: Union[str, Path], cfg: GeoConfig | None = None) -> GeoConfig:
    """Read a ``.geo`` wall-contour file into a :class:`GeoConfig`.

    If ``cfg`` is provided its non-geometry fields are preserved; only the
    wall arrays are (re)filled. Otherwise a default-valued config is
    returned with the geometry populated.
    """
    if cfg is None:
        cfg = GeoConfig()

    lines = Path(path).read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path}: .geo file appears truncated")

    n_z = int(float(lines[0].split()[0]))
    if n_z <= 0:
        raise ValueError(f"{path}: invalid number of axial stations ({n_z})")

    # line[1] is the column header "X R Xo Y0"; data starts at line[2]
    zs, rs, xs, ys = [], [], [], []
    for ln in lines[2:]:
        toks = ln.replace("\t", " ").split()
        if len(toks) < 4:
            continue
        zs.append(float(toks[0]))
        rs.append(float(toks[1]))
        xs.append(float(toks[2]))
        ys.append(float(toks[3]))
        if len(zs) == n_z:
            break

    cfg.z = np.array(zs, dtype=np.float64)
    cfg.r = np.array(rs, dtype=np.float64)
    cfg.xc = np.array(xs, dtype=np.float64)
    cfg.yc = np.array(ys, dtype=np.float64)
    return cfg
