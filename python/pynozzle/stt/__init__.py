"""Streamline Tracing Tool (port of STT2001).

The STT2001 program takes the streamline data emitted by ``MOC_Grid_BDE``
(``MOC_SL.plt``) plus the corresponding summary (``summary.out``) and grid
(``MOC_Grid.plt``) files, then for a chosen 3D throat shape:

1. finds the streamlines that pass through every angular station of the
   throat boundary (``CalcThroatSLs``),
2. trims those streamlines against any additional 3D constraint surfaces
   (``TrimSLs``, ``TrimSLsToMaxLength``, ``TrimSLsDueToAxiRevolution``),
3. resamples them onto an axial grid (``CalcGridSLs``), and
4. integrates wall pressure and surface area to give thrust / Isp / Cfg
   for the trimmed 3D nozzle (``CalcNozzleParameters``,
   ``GetPerformanceDataFromMOCSummaryFile``).

Public entry points mirror :mod:`pynozzle.moc2d`:

* :class:`pynozzle.stt.solver.STTSolver` — the main solver class
* :class:`pynozzle.stt.inp.STTInput` — ``.inp`` file reader/writer
* :func:`pynozzle.stt.cli.main` — command-line entry point
"""
from .solver import STTSolver, STTResult  # noqa: F401
from .inp import STTInput, read_inp, write_inp  # noqa: F401
