"""3D Method-of-Characteristics flow-field solver (port of ``3D_MOC``).

Given an axisymmetric (or laterally-offset) nozzle wall defined as a
stack of circles -- the ``.geo`` file written by ``MOC_Grid_BDE`` or a
streamline-traced contour -- plus the chamber/throat flow state, this
tool marches a full three-dimensional Method-of-Characteristics solution
plane by plane down the nozzle and writes the resulting mesh, wall, and
streamline data in Tecplot format.

The numerical method (Rice, JHU/APL): for each new axial plane the
solver locates interior "field points" and wall "body points" by
tracing bicharacteristics back to the previous plane, interpolating the
flow there with a thin-plate-spline surface fit, and solving the
compatibility + flow-tangency equations (the body point uses a
Newton-Raphson solve). The original used Numerical Recipes routines
(``newt``, ``ludcmp``/``lubksb``, ``sort2``); this port uses NumPy/SciPy
equivalents.

Public entry points:

* :class:`pynozzle.moc3d.solver.MOC3DGrid` — the solver class
* :class:`pynozzle.moc3d.geo.GeoConfig` — run configuration + ``.geo`` reader
* :func:`pynozzle.moc3d.cli.main` — command-line entry point
"""
from .solver import MOC3DGrid, MOC3DResult  # noqa: F401
from .geo import GeoConfig, read_geo  # noqa: F401
