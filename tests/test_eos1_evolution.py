"""Conservation and limiting-case checks for the EoS1 evolution correction."""

import itertools
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import odeint

from eom.massless_qgp_eom import MasslessQGPEoM
from eos.massless_qgp_eos import BETA1, BETA2, MasslessQGPEoS


class TestEoS1Evolution(unittest.TestCase):
    def setUp(self):
        self.eos = MasslessQGPEoS()
        self.model = MasslessQGPEoM(self.eos)

    def integrate(self, initial, rhos, ideal=False):
        solution, info = odeint(
            self.model.eom, initial, rhos, args=(ideal,),
            rtol=1e-10, atol=1e-12, full_output=True,
        )
        self.assertEqual(info["message"], "Integration successful.")
        self.assertTrue(np.isfinite(solution).all())
        return solution

    def test_conservation_equations(self):
        # Use EOS derivatives to check both conservation laws independently
        # of the expanded formula for F3, including nonzero shear.
        for T, ratio, shear, rho in itertools.product(
            (0.5, 1.2), (-15, -1, 0, 1, 10, 15), (-0.2, 0, 0.2), (-1, 0.7)
        ):
            with self.subTest(T=T, ratio=ratio, shear=shear, rho=rho):
                mu = np.array([ratio * T, 0., 0.])
                state = np.array([T, *mu, shear])
                dT = self.model.dT_drho(state, rho)
                dmu = self.model.dmu_drho(state, rho)[0]
                P = self.eos.pressure(T, mu)
                s = self.eos.entropy(T, mu)
                n = self.eos.number(T, mu)[0]
                h = np.tanh(rho)
                assert_allclose(s * dT + n * dmu,
                                -(4 / 3) * P * (2 - shear) * h,
                                rtol=1e-12, atol=1e-12)
                dn = (4 * BETA1 * mu[0] * T * dT
                      + (2 * BETA1 * T**2 + 12 * BETA2 * mu[0]**2) * dmu)
                assert_allclose(dn, -2 * n * h, rtol=1e-12, atol=1e-12)

    def test_zero_chemical_potential(self):
        for T, shear, rho in itertools.product((0.5, 1.2), (-0.2, 0, 0.2), (-1, 0.7)):
            with self.subTest(T=T, shear=shear, rho=rho):
                state = np.array([T, 0., 0., 0., shear])
                assert_allclose(self.model.f(T, 0.), 0.5, rtol=1e-14)
                assert_allclose(self.model.dT_drho(state, rho),
                                T * (shear - 2) * np.tanh(rho) / 3,
                                rtol=1e-12, atol=1e-12)
                assert_array_equal(self.model.dmu_drho(state, rho), np.zeros(3))

    def test_ideal_limit(self):
        # With initially zero shear and ideal_evolution=True, both T and mu
        # have the exact cosh(rho)^(-2/3) solution.
        for ratio, direction in itertools.product((-15, 0, 15), (-1, 1)):
            with self.subTest(ratio=ratio, direction=direction):
                rhos = np.linspace(0, direction * 3, 121)
                initial = np.array([1.2, ratio * 1.2, 0., 0., 0.])
                solution = self.integrate(initial, rhos, ideal=True)
                expected = np.cosh(rhos)[:, None] ** (-2 / 3) * initial[None, :2]
                assert_allclose(solution[:, :2], expected, rtol=5e-9, atol=1e-10)
                assert_array_equal(solution[:, 2:], np.zeros((len(rhos), 3)))

    def test_baryon_conservation_on_trajectories(self):
        # Match the figure scripts' initial mu_B/T values and integrate in
        # both directions with the full shear-relaxation equation enabled.
        for ratio, direction in itertools.product((0, 10, 15), (-1, 1)):
            with self.subTest(ratio=ratio, direction=direction):
                rhos = np.linspace(0, direction * 3, 121)
                initial = np.array([1.2, ratio * 1.2, 0., 0., 0.])
                solution = self.integrate(initial, rhos)
                n = np.array([self.eos.number(y[0], y[1:4])[0] for y in solution])
                assert_allclose(n * np.cosh(rhos)**2, n[0], rtol=1e-8, atol=1e-10)
                assert_array_equal(solution[:, 2:4], np.zeros((len(rhos), 2)))
                if ratio == 0:
                    assert_array_equal(solution[:, 1], np.zeros(len(rhos)))


if __name__ == "__main__":
    unittest.main()
