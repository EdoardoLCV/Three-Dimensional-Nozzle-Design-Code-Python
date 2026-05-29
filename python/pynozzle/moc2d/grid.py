"""Data container for the MOC grid -- mirrors the private array members of
the C++ ``MOC_GridCalc`` class.

In the original code, every per-point quantity is a ``double **`` indexed as
``mach[i][j]`` etc., where ``i`` is the LRC index (wall = 0, increases toward
centerline) and ``j`` is the RRC index (throat = 0, increases downstream).
For each RRC ``j``, the centerline lives at ``i = iLast[j]``.

Here we use NumPy 2D arrays with shape ``(maxLRC, maxRRC)``. The semantics
of indexing (``arr[i, j]``) and of ``i_last[j]`` are unchanged.
"""
from __future__ import annotations

import numpy as np


class MOCGrid:
    """Holds all per-point arrays plus the LRC-DE arrays.

    Parameters
    ----------
    max_lrc, max_rrc
        Array shape. Defaults match the C++ defaults
        (``maxLRC = 1000``, ``maxRRC = 999``).
    """

    __slots__ = (
        "max_lrc",
        "max_rrc",
        "mach",
        "pres",
        "temp",
        "rho",
        "gamma",
        "x",
        "r",
        "theta",
        "massflow",
        "thrust",
        "s_thrust",
        "i_last",
        # LRC-DE 1-D arrays
        "x_de",
        "r_de",
        "m_de",
        "p_de",
        "t_de",
        "rho_de",
        "g_de",
        "theta_de",
        "mass_de",
    )

    def __init__(self, max_lrc: int = 1000, max_rrc: int = 999) -> None:
        self.max_lrc = max_lrc
        self.max_rrc = max_rrc
        shape = (max_lrc, max_rrc)
        self.mach     = np.zeros(shape)
        self.pres     = np.zeros(shape)
        self.temp     = np.zeros(shape)
        self.rho      = np.zeros(shape)
        self.gamma    = np.zeros(shape)
        self.x        = np.zeros(shape)
        self.r        = np.zeros(shape)
        self.theta    = np.zeros(shape)
        self.massflow = np.zeros(shape)
        self.thrust   = np.zeros(shape)
        self.s_thrust = np.zeros(shape)
        self.i_last   = np.zeros(max_lrc, dtype=np.int64)
        # LRC-DE arrays -- one entry per RRC index
        self.x_de     = np.zeros(max_lrc)
        self.r_de     = np.zeros(max_lrc)
        self.m_de     = np.zeros(max_lrc)
        self.p_de     = np.zeros(max_lrc)
        self.t_de     = np.zeros(max_lrc)
        self.rho_de   = np.zeros(max_lrc)
        self.g_de     = np.zeros(max_lrc)
        self.theta_de = np.zeros(max_lrc)
        self.mass_de  = np.zeros(max_lrc)

    # ------------------------------------------------------------------
    def reset(self, i_start: int, i_end: int, j_start: int, j_end: int) -> None:
        """Zero out a rectangular sub-region of the grid.

        Direct port of ``MOC_GridCalc::ResetGrid``. Used after the iterative
        thetaB search to clear out stale data left by a previous trial.
        """
        i_end = min(i_end, self.max_lrc - 1)
        j_end = min(j_end, self.max_rrc - 1)
        s_i = slice(i_start, i_end + 1)
        s_j = slice(j_start, j_end + 1)
        self.mach[s_i, s_j]     = 0.0
        self.pres[s_i, s_j]     = 0.0
        self.temp[s_i, s_j]     = 0.0
        self.rho[s_i, s_j]      = 0.0
        self.gamma[s_i, s_j]    = 0.0
        self.x[s_i, s_j]        = 0.0
        self.r[s_i, s_j]        = 0.0
        self.theta[s_i, s_j]    = 0.0
        self.massflow[s_i, s_j] = 0.0
        self.thrust[s_i, s_j]   = 0.0
        self.s_thrust[s_i, s_j] = 0.0
