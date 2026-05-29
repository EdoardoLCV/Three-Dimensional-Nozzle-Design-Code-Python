"""2D / axisymmetric Method-of-Characteristics nozzle design tool.

Direct port of the ``MOC_Grid_BDE`` MFC application from the original
JHU/APL distribution.

Public entry points:

* :class:`pynozzle.moc2d.solver.MOCGridCalc` - the solver, equivalent to the
  C++ ``MOC_GridCalc`` class.
* :class:`pynozzle.moc2d.inp.MOCInput` - parser for the ``.inp`` files written
  by the original GUI.
* :func:`pynozzle.moc2d.cli.main` - command-line entry point.
"""
from .solver import MOCGridCalc, MOCResult  # noqa: F401
from .inp import MOCInput, read_inp, write_inp  # noqa: F401
