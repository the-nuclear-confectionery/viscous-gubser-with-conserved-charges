from numpy import tanh, pi, array, sqrt, fabs, where
import numpy as np
from scipy.interpolate import interp1d
from utils.constants import HBARC
from utils.analytic_functions import rho, u_x, u_y
from eom.base_eom import BaseEoM
from typing import Union, Dict

class MasslessQGPEoM(BaseEoM):
    """
    Implements the equations of motion for the massless QGP.
    Expects the state vector ys = [T, μ_B, μ_S, μ_Q, π_hat]. For this system,
    μ_S and μ_Q are assumed to be zero throughout the evolution.
    """
    def __init__(self, eos_instance, eom_params: Dict[str, float] = {'CTAUR': 5, 'ETA_OVER_S': 0.2}):
        self.eos = eos_instance
        self.CTAUR = eom_params['CTAUR']
        self.ETA_OVER_S = eom_params['ETA_OVER_S']

    def f(self, T, muB):
        # Dummy placeholder function; replace with the proper expression.
        return 1.0

    def tau_R(self, T, muB):
        # τ_R = CTAUR * ETA_S * s / (e + p)
        e = self.eos.energy(T, muB)
        p = self.eos.pressure(T, muB)
        s = self.eos.entropy(T, muB)
        return self.CTAUR * self.ETA_OVER_S * s / (e + p)

    def dT_drho(self, ys: np.ndarray, rho_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        T, muB, muS, muQ, pi_hat = ys
        mu = np.array([muB, muS, muQ])
        f_val = self.f(T, muB)
        deriv = 1 + (4 * muB**2) / (np.pi**2 * T**2)
        deriv *= -f_val * pi_hat
        deriv += 1
        return -(2/3) * T * deriv * np.tanh(rho_val)

    def dmu_drho(self, ys: np.ndarray, rho_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        T, muB, muS, muQ, pi_hat = ys
        f_val = self.f(T, muB)
        # Only μ_B evolves in the massless QGP case.
        dmuB = -(2/3) * muB * (1 + 2 * f_val * pi_hat) * np.tanh(rho_val)
        # μ_S and μ_Q remain zero.
        return np.array([dmuB, 0.0, 0.0])

    def dpi_drho(self, ys: np.ndarray, rho_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        T, muB, muS, muQ, pi_hat = ys
        tau_r = self.tau_R(T, muB)
        return (4/(3*self.CTAUR)) * np.tanh(rho_val) - pi_hat / tau_r - (4/3) * pi_hat**2 * np.tanh(rho_val)


    # --- Milne conversion functions for massless_qgp ---
    @staticmethod
    def milne_energy(tau: float,
                    x: Union[float, np.ndarray],
                    y: Union[float, np.ndarray],
                    q: float,
                    ads_T: interp1d,
                    ads_mu: list,
                    temperature_0: float,
                    chem_potential_0: np.ndarray,
                    # eom_instance: MasslessQGP_EoM,
                    eos,
                    tol: float = 1e-20) -> Union[float, np.ndarray]:
        r = sqrt(x**2 + y**2)
        rh = rho(tau, r, q)
        temp = ads_T(rh)
        mu = array([f(rh) for f in ads_mu])
        if isinstance(temp, np.ndarray):
            energy_val = eos.energy(temp, mu)
            return HBARC * energy_val / tau**4
        else:
            if temp <= tol:
                temp = tol
            energy_val = eos.energy(temp, mu)
            val = HBARC * energy_val / tau**4
            return tol if val < tol else val

    @staticmethod
    def milne_number(tau: float,
                    x: Union[float, np.ndarray],
                    y: Union[float, np.ndarray],
                    q: float,
                    ads_T: interp1d,
                    ads_mu: list,
                    temperature_0: float,
                    chem_potential_0: np.ndarray,
                    # eom_instance: MasslessQGP_EoM,
                    eos,
                    tol: float = 1e-20) -> Union[float, np.ndarray]:
        r = sqrt(x**2 + y**2)
        rh = rho(tau, r, q)
        temp = ads_T(rh)
        mu = array([f(rh) for f in ads_mu])
        if isinstance(temp, np.ndarray):
            number_val = eos.number(temp, mu)
            return number_val / tau**3
        else:
            if temp <= tol:
                temp = tol
            number_val = eos.number(temp, mu)
            n = number_val / tau**3
            n[where(n < tol)] = tol
            return n

    @staticmethod
    def milne_entropy(tau: float,
                    x: Union[float, np.ndarray],
                    y: Union[float, np.ndarray],
                    q: float,
                    ads_T: interp1d,
                    ads_mu: list,
                    temperature_0: float,
                    chem_potential_0: np.ndarray,
                    # eom_instance: MasslessQGP_EoM,
                    eos,
                    tol: float = 1e-20) -> Union[float, np.ndarray]:
        r = sqrt(x**2 + y**2)
        rh = rho(tau, r, q)
        temp = ads_T(rh)
        mu = array([f(rh) for f in ads_mu])
        if isinstance(temp, np.ndarray):
            entropy_val = eos.entropy(temp, mu)
            return entropy_val / tau**3
        else:
            if temp <= tol:
                temp = tol
            entropy_val = eos.entropy(temp, mu)
            s = entropy_val / tau**3
            return tol if s < tol else s

    @staticmethod
    def milne_pi(tau: float,
                x: Union[float, np.ndarray],
                y: Union[float, np.ndarray],
                q: float,
                ads_T: interp1d,
                ads_mu: list,
                ads_pi_bar_hat: interp1d,
                temperature_0: float,
                chem_potential_0: np.ndarray,
                # eom_instance: MasslessQGP_EoM,
                eos,
                tol: float = 1e-20,
                nonzero_xy: bool = False
                ) -> list:
        r = sqrt(x**2 + y**2)
        rh = rho(tau, r, q)
        temp = ads_T(rh)
        mu = array([f(rh) for f in ads_mu])
        if not isinstance(temp, np.ndarray):
            if temp <= tol:
                temp = tol
        e_val = eos.energy(temp, mu)
        p_val = eos.pressure(temp, mu)
        pi_hat = HBARC * (e_val + p_val) * ads_pi_bar_hat(rh)
        pi_nn = pi_hat / tau**6
        pi_xx = -0.5 * (1 + u_x(tau, x, y, q)**2) * pi_hat / tau**4
        pi_yy = -0.5 * (1 + u_y(tau, x, y, q)**2) * pi_hat / tau**4
        if nonzero_xy:
            y = x
        pi_xy = -0.5 * u_x(tau, x, y, q) * u_y(tau, x, y, q) * pi_hat / tau**4
        if not isinstance(temp, np.ndarray):
            pi_nn = tol if fabs(pi_nn) < tol else pi_nn
            pi_xx = tol if fabs(pi_xx) < tol else pi_xx
            pi_yy = tol if fabs(pi_yy) < tol else pi_yy
            pi_xy = tol if fabs(pi_xy) < tol else pi_xy
        return [pi_xx, pi_yy, pi_xy, pi_nn]