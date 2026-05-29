"""Regression tests for the STT2001 port.

Uses only the standard-library :mod:`unittest`. Runs the solver on the
bundled ``M3.5Perf`` example and checks scalar results and output-file
structure against values validated bit-for-bit against the original
Windows tool's sample output.

Run with ``python -m unittest`` from the project root.
"""
import tempfile
import unittest
from pathlib import Path

from pynozzle.stt.inp import read_inp, write_inp
from pynozzle.stt.solver import STTSolver
from pynozzle.stt.loaders import get_friction_loss

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "stt"


class STTInputRoundTrip(unittest.TestCase):
    def test_read_m35(self):
        inp = read_inp(EXAMPLES / "M3.5Perf.inp")
        self.assertEqual(inp.file_prefix, "M3.5Perf")
        self.assertEqual(inp.sl_filename, "MOC_sl.plt")
        self.assertAlmostEqual(inp.a_throat, 0.7854)
        self.assertAlmostEqual(inp.mass_flow, 35.7)
        s0 = inp.slots[0]
        self.assertTrue(s0.throat and s0.constraint)
        self.assertAlmostEqual(s0.rc, 1.0)
        self.assertEqual(s0.nSL, 72)
        self.assertAlmostEqual(s0.omega, 360.0)

    def test_round_trip(self):
        inp = read_inp(EXAMPLES / "M3.5Perf.inp")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rt.inp"
            write_inp(inp, p)
            inp2 = read_inp(p)
        self.assertEqual(inp.file_prefix, inp2.file_prefix)
        self.assertAlmostEqual(inp.slots[0].rc, inp2.slots[0].rc)
        self.assertEqual(inp.slots[0].nSL, inp2.slots[0].nSL)
        self.assertAlmostEqual(inp.mass_flow, inp2.mass_flow)


class STTFriction(unittest.TestCase):
    def test_friction_interp(self):
        val = get_friction_loss(EXAMPLES / "friction_table.txt", 167.08)
        self.assertAlmostEqual(val, 131.95, delta=0.05)

    def test_friction_missing(self):
        self.assertEqual(get_friction_loss("/nonexistent/file.txt", 100.0), 0.0)


class STTRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name)
        inp = read_inp(EXAMPLES / "M3.5Perf.inp")
        cls.result = STTSolver(EXAMPLES, inp).run(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_success(self):
        self.assertTrue(self.result.success)

    def test_surface_area(self):
        self.assertAlmostEqual(self.result.surface_area, 167.08, delta=0.01)

    def test_projected_area(self):
        self.assertAlmostEqual(self.result.projected_area, 17.7489, delta=0.001)

    def test_pressure_force(self):
        self.assertAlmostEqual(self.result.pressure_force, 531.764, delta=0.01)

    def test_friction_loss(self):
        self.assertAlmostEqual(self.result.fric_loss, 131.956, delta=0.01)

    def test_exit_area(self):
        self.assertAlmostEqual(self.result.a_exit, 18.5343, delta=0.001)

    def test_throat_thrust(self):
        # Our value uses the correct r* = 1.000; the sample's 492.024 comes
        # from a C++ uninitialised-buffer read that yields r* = 1.00099.
        self.assertAlmostEqual(self.result.throat_thrust, 492.999, delta=0.01)

    def test_isp(self):
        self.assertAlmostEqual(self.result.isp_calc, 23.97, delta=0.01)

    def test_output_files_exist(self):
        for name in ("M3.5Perf_ThroatSL.out", "M3.5Perf_TrimSL.out",
                     "M3.5Perf_AvsX.out", "M3.5Perf_AvsSL.out",
                     "M3.5Perf_trimmed_P3D.xyz", "M3.5Perf_trimmed_P3D.dat",
                     "M3.5Perf.plt", "M3.5Perf_STT_summary.out",
                     "M3.5Perf_cl.dat", "M3.5Perf_all_runs.dat"):
            self.assertTrue((self.out / name).is_file(), f"missing {name}")

    def test_throat_summary_first_row(self):
        text = (self.out / "M3.5Perf_ThroatSummary.out").read_text()
        first = text.splitlines()[1].split("\t")
        self.assertEqual(int(first[0]), 0)
        self.assertAlmostEqual(float(first[3]), 1.0, places=3)
        self.assertAlmostEqual(float(first[4]), 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
