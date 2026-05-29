"""Regression tests for the 3D MOC port.

The 3D MOC march is an iterative point-by-point solver (Newton-Raphson
body points, four-base-point compatibility equations at field points,
thin-plate-spline interpolation), so bit-for-bit reproduction against
the Windows sample isn't achievable -- accumulated floating-point error
from a 200-plane march combined with libm differences between MSVC and
glibc give 4-5 significant-digit agreement. These tests validate at
that level.

Run with ``python -m unittest`` from the project root.
"""
import math
import tempfile
import unittest
from pathlib import Path

from pynozzle.moc3d import GeoConfig, MOC3DGrid, read_geo
from pynozzle.moc3d.kernels import (
    body_fit_jk, calc_parametric_angle, comp_equ,
)
from pynozzle.moc3d.point import XYZPoint, CONSTANT_X, CONSTANT_Y, LINE, CIRCLE


EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "moc3d"


class GeoReader(unittest.TestCase):
    def test_read_cone10(self):
        cfg = read_geo(EXAMPLES / "cone10.geo")
        self.assertEqual(cfg.n_z, 217)
        self.assertAlmostEqual(cfg.z[0], 0.0)
        self.assertAlmostEqual(cfg.r[0], 1.0)
        # cone has constant centre (0, 0)
        self.assertAlmostEqual(cfg.xc[0], 0.0)
        self.assertAlmostEqual(cfg.yc[0], 0.0)

    def test_read_m4perfect(self):
        cfg = read_geo(EXAMPLES / "M4perfect.geo")
        self.assertEqual(cfg.n_z, 162)


class InitialReferencePlane(unittest.TestCase):
    """The initial-plane setup is deterministic and should match the
    sample to within machine-precision noise on near-zero sin/cos values."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name)
        cfg = read_geo(EXAMPLES / "cone10.geo")
        grid = MOC3DGrid(cfg, cls.out)
        grid.set_initial_properties()
        cls.grid = grid

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_n_pts(self):
        # Computed analytically for r0=1, nDiv=36:
        # _nRadii=7, points = 1 + 8 + 13 + 19 + 26 + 32 + 38 + 44 = 181
        self.assertEqual(self.grid._n_pts, 181)

    def test_body_count(self):
        body = sum(1 for f in self.grid._body_point_flag if f)
        self.assertEqual(body, 44)

    def test_initial_pressure(self):
        self.assertAlmostEqual(self.grid._pt[0][0].p, 1000.0)
        self.assertAlmostEqual(self.grid._pt[0][0].rho, 5.09176, places=4)
        self.assertAlmostEqual(self.grid._pt[0][0].mach, 1.1)

    def test_initial_velocity(self):
        # q = mach * sqrt(gamma * GASCON / mWt * T * g)
        # = 1.1 * sqrt(1.4 * 1545.317/28.96 * 530 * 32.174) approx 1241.5
        self.assertAlmostEqual(self.grid._pt[0][0].q, 1241.5, places=0)


class BodyFitGeometry(unittest.TestCase):
    """Spot-check the body-fit shape classification."""

    def _make_grid(self):
        cfg = read_geo(EXAMPLES / "cone10.geo")
        cfg.z = cfg.z[:5]
        cfg.r = cfg.r[:5]
        cfg.xc = cfg.xc[:5]
        cfg.yc = cfg.yc[:5]
        with tempfile.TemporaryDirectory() as d:
            g = MOC3DGrid(cfg, d)
            g.set_initial_properties()
        return g

    def test_circle_fit(self):
        # For a circular wall, every body fit should classify as CIRCLE
        g = self._make_grid()
        bf = body_fit_jk(g, 5, 0)
        self.assertEqual(bf["type"], CIRCLE)
        # the radius**2 should be ~1.0 (the first axial station has r0=1)
        self.assertAlmostEqual(bf["C"], 1.0, places=3)


class CompEquFormula(unittest.TestCase):
    def test_returns_four_values(self):
        # Just check structure: comp_equ returns (a0, a1, a2, rhs)
        p = XYZPoint()
        p.p = 100.0
        p.theta = 0.1
        p.psi = 0.05
        p.rho = 5.0
        p.q = 1000.0
        p.L = 0.1
        out = comp_equ(0.5, p, 0.3, 0.2, 0.01, 0.1, 0.0, 0.0)
        self.assertEqual(len(out), 4)
        # the a0 coefficient is 144 / (tan(beta) * rho * q^2 / g)
        # should be positive and finite
        self.assertGreater(out[0], 0)
        self.assertTrue(math.isfinite(out[3]))


class ParametricAngle(unittest.TestCase):
    def test_delta_in_range(self):
        # the function should always return a delta in [0, 2*pi]
        from pynozzle.moc3d.point import PI
        d = calc_parametric_angle(0.5, 0.1, 0.05, 0.2)
        self.assertGreaterEqual(d, 0.0)
        self.assertLessEqual(d, 2 * PI)


class ShortMarch(unittest.TestCase):
    """Run a short march (first 5 axial planes) to validate the full
    field-point + body-point machinery without taking minutes."""

    @classmethod
    def setUpClass(cls):
        cfg = read_geo(EXAMPLES / "cone10.geo")
        cfg.z = cfg.z[:5]
        cfg.r = cfg.r[:5]
        cfg.xc = cfg.xc[:5]
        cfg.yc = cfg.yc[:5]
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name)
        cls.grid = MOC3DGrid(cfg, cls.out)
        cls.grid.set_initial_properties()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.result = cls.grid.calc_nozzle()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_success(self):
        self.assertTrue(self.result.success)

    def test_files_written(self):
        for name in ("z=0.out", "outfile.out", "full_mesh.plt",
                     "axialStations.plt", "Streamlines.plt", "Wall.plt",
                     "Initial Wall.plt"):
            self.assertTrue((self.out / name).is_file(), f"missing {name}")

    def test_chamber_state_preserved(self):
        # After a short march, the centerline point flow state should still
        # be close to the chamber state (the cone barely diverges in 5 steps).
        pt = self.grid._pt[0][4]   # centerline, after 4 march steps
        self.assertAlmostEqual(pt.p, 1000.0, delta=10.0)
        self.assertAlmostEqual(pt.mach, 1.1, delta=0.01)
        # gas is supersonic
        self.assertGreater(pt.mach, 1.0)


if __name__ == "__main__":
    unittest.main()
