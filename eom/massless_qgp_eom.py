from numpy import tanh, pi, array, sqrt, fabs, where
import numpy as np
from scipy.interpolate import interp1d
from utils.constants import HBARC
from utils.analytic_functions import rho, u_x, u_y
from eos.massless_qgp_eos import *
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

    # Eq. (21) in the paper
    def f(self, T, muB):
        numerator_A = 36 * (np.pi ** 2) * gQCD * (T ** 4)
        numerator_B = 5 * gq * ((3 * np.pi ** 2) * (muB ** 2) * (T ** 2) + 2 * muB ** 4)
        denominator_A = 72 * (np.pi ** 2) * gQCD * (T ** 2) * ((np.pi ** 2) * (T ** 2) + 4 * muB ** 2)
        denominator_B = - 5 * gq * (muB ** 2) * (3 * np.pi ** 2 * T ** 2 - 4 * muB ** 2)
        numerator = numerator_A + numerator_B
        denominator = denominator_A + denominator_B
        return numerator / denominator

    # Eq. (16) in the paper
    def tau_R(self, T, mu):
        e = self.eos.energy(T, mu)
        p = self.eos.pressure(T, mu)
        s = self.eos.entropy(T, mu)
        return self.CTAUR * self.ETA_OVER_S * s / (e + p)

    # Eq. (22a) in the paper
    def dT_drho(self, ys: np.ndarray, rho_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        T_hat, muB_hat, muS_hat, muQ_hat, pi_hat = ys
        mu_hat = np.array([muB_hat, muS_hat, muQ_hat])
        ret_val = 1 + (4 / np.pi ** 2) * ((muB_hat / T_hat) ** 2)
        ret_val = 1 - ret_val * self.f(T_hat, muB_hat) * pi_hat
        ret_val *= (- 2 / 3) * np.tanh(rho_value) * T_hat
        return ret_val

    # Eq. (22b) in the paper
    def dmu_drho(self, ys: np.ndarray, rho_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        T_hat, muB_hat, muS_hat, muQ_hat, pi_hat = ys
        mu_hat = np.array([muB_hat, muS_hat, muQ_hat])
        ret_val = (- 2 / 3) * (1 + 2 * self.f(T_hat, muB_hat) * pi_hat) * np.tanh(rho_value) * muB_hat
        return ret_val * np.array([1, 0, 0])

    # Eq. (20c) in the paper
    def dpi_drho(self, ys: np.ndarray, rho_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        T_hat, muB_hat, muS_hat, muQ_hat, pi_hat = ys
        mu_hat = np.array([muB_hat, muS_hat, muQ_hat])
        tau_R = self.tau_R(T_hat, mu_hat)
        ret_val = (4 / (3 * self.CTAUR)) * np.tanh(rho_value)
        ret_val -= pi_hat / tau_R
        ret_val -= (4 / 3) * pi_hat ** 2 * np.tanh(rho_value)
        return ret_val