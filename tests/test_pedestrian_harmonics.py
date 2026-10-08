"""Regression tests for pedestrian harmonic frequencies and phases.

Run from the repository root with: python -m unittest discover -s tests -v
Uses the same NumPy, SciPy, Matplotlib and imageio dependencies as MDyn.
"""

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import numpy as np
from MDyn_forcing import MovingLoads


class PedestrianHarmonicsTests(unittest.TestCase):
    def make_load(self, amplitudes, phases=None):
        return MovingLoads(
            DOFactive=[3, 4],
            DirectionLoad=3,
            DirectionMoment=4,
            NumberOfNodes=2,
            NodeX=np.array([0.0, 1.0]),
            BeamNumber=np.array([1]),
            VehicleBeams=np.array([[1, 0]]),
            XVehicle=[[0.0]],
            YVehicle=[[0.0]],
            Gv=[amplitudes],
            phiv=[phases if phases is not None else [0.0] * len(amplitudes)],
            fmv=[2.0],
        )

    def test_each_harmonic_reaches_extrema_at_its_own_quarter_periods(self):
        for order in (1, 2, 3):
            with self.subTest(order=order):
                amplitudes = [0.0, 0.0, 0.0, 0.0]
                amplitudes[order] = 70.0
                load = self.make_load(amplitudes)
                period = 1.0 / (order * 2.0)
                for fraction, expected in ((0.25, 70.0), (0.75, -70.0)):
                    _, force = load.SteppingMovingLoads(0, [0.5, 0.0], fraction * period)
                    self.assertAlmostEqual(force, expected)

    def test_phase_applies_to_first_harmonic(self):
        load = self.make_load([700.0, 280.0], [0.0, np.pi / 2])
        _, force = load.SteppingMovingLoads(0, [0.5, 0.0], 0.0)
        self.assertAlmostEqual(force, 420.0)

    def test_three_harmonic_load_and_nodal_resultants(self):
        load = self.make_load(
            [700.0, 280.0, 70.0, 70.0], [0.0, 0.0, np.pi / 2, np.pi / 2]
        )
        # At t=1/8 s, the three sine contributions are 280, 70 and 0 N.
        nodal, force = load.SteppingMovingLoads(0, [0.25, 0.2], 0.125)
        self.assertAlmostEqual(force, 1050.0)
        np.testing.assert_allclose(nodal, [787.5, 262.5, 157.5, 52.5])

    def test_single_zero_phase_harmonic_preserves_signed_load(self):
        load = self.make_load([-700.0, -280.0])
        _, force = load.SteppingMovingLoads(0, [0.5, 0.0], 0.125)
        self.assertAlmostEqual(force, -980.0)

    def test_load_outside_bridge_is_zero(self):
        load = self.make_load([700.0, 280.0])
        for x in (-0.1, 1.1):
            self.assertEqual(load.SteppingMovingLoads(0, [x, 0.0], 0.125), [0, 0])


if __name__ == "__main__":
    unittest.main()
