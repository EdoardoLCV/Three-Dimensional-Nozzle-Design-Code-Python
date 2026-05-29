"""Regression and smoke tests for :mod:`pynozzle.moc2d`.

Uses only the standard-library :mod:`unittest`, so it runs anywhere with
``python -m unittest`` from the project root.

The sample-matching tests need the original C++ output directory. They
look first at ``$PYNOZZLE_SAMPLE_DIR``, then at the upstream
distribution path, and skip if neither exists. The scalar-extraction
test runs unconditionally.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from pynozzle.moc2d import read_inp
from pynozzle.moc2d.cli import _build_calc
from pynozzle.moc2d.io_writers import write_all


SAMPLE_ENV = "PYNOZZLE_SAMPLE_DIR"
DEFAULT_SAMPLE = Path(
    "/home/claude/work/Three-Dimensional-Nozzle-Design-Code/"
    "MOC_Grid_BDE/outputs_M3.5Perf"
)
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _sample_dir():
    path = Path(os.environ.get(SAMPLE_ENV, DEFAULT_SAMPLE))
    return path if (path / "summary.out").is_file() else None


def _strip_cr(path):
    return [ln.rstrip("\r\n")
            for ln in path.read_text(errors="replace").splitlines()]


class _M35Mixin:
    """Shared setUp: solve the M3.5Perf case once into a temp directory."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out_dir = Path(cls._tmp.name)
        inp = read_inp(EXAMPLES / "M3.5Perf.inp")
        calc = _build_calc(inp, full_output=True)
        result = calc.run()
        if not result.success:
            raise AssertionError(result.error_message or "Solver failed")
        write_all(result, cls.out_dir, full_output=True)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


class TestM35MatchesSample(_M35Mixin, unittest.TestCase):
    """Diff the full M3.5 outputs against the upstream sample."""

    def test_summary_matches(self):
        sample = _sample_dir()
        if sample is None:
            self.skipTest(
                f"Sample dir not present; set ${SAMPLE_ENV} or run from the "
                "upstream distribution."
            )
        ours = _strip_cr(self.out_dir / "summary.out")
        theirs = _strip_cr(sample / "summary.out")
        diffs = sum(1 for a, b in zip(ours, theirs) if a != b)
        # Two documented cosmetic differences:
        #   - "Massflow error..." line: C++ has uninitialized memory
        #   - One trailing zero-vs-1e-16 floating-point difference
        self.assertLessEqual(diffs, 2,
            f"{diffs} unexpected mismatches with sample summary.out")

    def test_rao_dat_matches(self):
        sample = _sample_dir()
        if sample is None:
            self.skipTest(f"Sample dir not present; set ${SAMPLE_ENV} to run.")
        ours = _strip_cr(self.out_dir / "rao.dat")
        theirs = _strip_cr(sample / "rao.dat")
        self.assertEqual(ours, theirs)


class TestM35Scalars(_M35Mixin, unittest.TestCase):
    """Self-contained scalar checks -- runs without the sample dir."""

    def _grab(self, key):
        text = (self.out_dir / "summary.out").read_text()
        for ln in text.splitlines():
            if ln.startswith(key):
                return ln.split("\t", 1)[1].strip()
        raise AssertionError(f"missing key in summary: {key}")

    def test_theta_b(self):
        self.assertEqual(self._grab("Initial Expansion Angle ThetaB(deg):"),
                         "15.2196")

    def test_length(self):
        self.assertEqual(self._grab("Nozzle Length/R*:"), "12.5363")

    def test_expansion(self):
        self.assertEqual(self._grab("Expansion Ratio:"), "6.73651")

    def test_mdot(self):
        self.assertEqual(self._grab("Massflow (lbm/s):"), "35.9981")

    def test_thrust(self):
        self.assertEqual(self._grab("Stream Thrust (lbf):"), "2518.05")

    def test_isp(self):
        self.assertEqual(self._grab("Isp (lbf-s/lbm):"), "69.9496")


class TestSmoke(unittest.TestCase):
    """Smoke tests for the Cone and Rao solvers."""

    def test_cone_runs(self):
        inp = read_inp(EXAMPLES / "cone10.inp")
        calc = _build_calc(inp, full_output=False)
        result = calc.run()
        self.assertTrue(result.success, result.error_message)
        m_exit = result.grid.mach[0, result.last_rrc]
        self.assertLess(abs(m_exit - 4.0), 0.01)

    def test_rao_runs(self):
        inp = read_inp(EXAMPLES / "M4Rao.inp")
        calc = _build_calc(inp, full_output=False)
        result = calc.run()
        self.assertTrue(result.success, result.error_message)
        m_exit = result.grid.mach[0, result.last_rrc]
        self.assertLess(abs(m_exit - 4.0), 0.01)
        self.assertGreater(result.theta_b_ans, 0.4)


if __name__ == "__main__":
    unittest.main()
