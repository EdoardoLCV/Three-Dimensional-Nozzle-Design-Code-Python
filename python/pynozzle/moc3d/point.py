"""Point class and physical constants for the 3D MOC solver.

The C++ used a small ``XYZPoint`` class with a flat set of doubles; we
mirror it with ``__slots__`` for speed and low memory (there can be
tens of thousands of points across all planes).

Note the gas constant: the 3D tool uses ``GASCON = 1545.317`` (slightly
different from the value baked into the 2D tool's constants), so we
define it locally to keep the 3D results faithful.
"""
from __future__ import annotations

import math

# Physical constants (match engineering_constants.hpp used by 3D_MOC)
PI = math.acos(-1.0)
DEG_PER_RAD = 180.0 / PI
RAD_PER_DEG = 1.0 / DEG_PER_RAD
GRAV = 32.17400
GASCON = 1545.317   # ft-lbf / (lbmol-R) -- the value 3D_MOC uses

# bodyType enum
CONSTANT_X = 0
CONSTANT_Y = 1
LINE = 2
CIRCLE = 3

# secantMethod / failflag enums
FAIL = 0
FAIL_MACH = 1
FAIL_HIGH = 2
OK = 3
NO = 0
YES = 1


class XYZPoint:
    """A single MOC grid point (port of the C++ ``XYZPoint``).

    Holds geometry (``x``, ``y``, ``z``) plus flow state (pressure,
    temperature, density, Mach, velocity, gamma, molecular weight, the
    two flow angles ``theta``/``psi``, the bicharacteristic length
    ``L`` and parametric angle ``delta``).
    """
    __slots__ = ("x", "y", "z", "p", "t", "mach", "g", "mol_wt", "rho",
                 "theta", "q", "psi", "L", "delta")

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.p = 0.0
        self.t = 0.0
        self.mach = 0.0
        self.g = 0.0
        self.mol_wt = 0.0
        self.rho = 0.0
        self.theta = 0.0
        self.q = 0.0
        self.psi = 0.0
        self.L = 0.0
        self.delta = 0.0

    def copy_from(self, other: "XYZPoint") -> None:
        for s in self.__slots__:
            setattr(self, s, getattr(other, s))

    def clone(self) -> "XYZPoint":
        p = XYZPoint()
        p.copy_from(self)
        return p
