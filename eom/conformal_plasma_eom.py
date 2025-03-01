import numpy as np
from scipy.interpolate import interp1d
from utils.constants import HBARC
from utils.analytic_functions import rho, u_x, u_y
from eos.base_eos import BaseEoS
from eom.base_eom import BaseEoM
from typing import Union, Optional, Dict
from eos.conformal_plasma_eos import ConformalPlasmaEoS
from eos.conformal_plasma_eos import ALPHA

class ConformalPlasmaEoM(BaseEoM):
    """
    Implements the equations of motion for the conformal plasma.
    Expects the state vector ys = [T, μ_B, μ_S, μ_Q, π_hat]. All three chemical potentials are allowed.
    """
    def __init__(self, eos_instance, eos_params: Dict[str, Union[float, np.ndarray]], eom_params: Dict[str, float] = {'CTAUR': 5, 'ETA_OVER_S': 0.2}):
        self.eos = eos_instance
        self.T_ast = eos_params['T_ast']
        self.mu_ast = eos_params['mu_ast']
        self.CTAUR = eom_params['CTAUR']
        self.ETA_OVER_S = eom_params['ETA_OVER_S']

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
        ret_val = (1 / 3) * (pi_hat - 2) + pi_hat * np.sum(((self.T_ast * mu_hat) / (T_hat * self.mu_ast)) ** 2)
        ret_val *= T_hat * np.tanh(rho_value)
        return ret_val

    # Eq. (22b) in the paper
    def dmu_drho(self, ys: np.ndarray, rho_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        T_hat, muB_hat, muS_hat, muQ_hat, pi_hat = ys
        mu_hat = np.array([muB_hat, muS_hat, muQ_hat])
        ret_val = -(2 / 3) * mu_hat * (1 + pi_hat) * np.tanh(rho_value)
        return ret_val

    # Eq. (20c) in the paper
    def dpi_drho(self, ys: np.ndarray, rho_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        T_hat, muB_hat, muS_hat, muQ_hat, pi_hat = ys
        mu_hat = np.array([muB_hat, muS_hat, muQ_hat])
        tau_R = self.tau_R(T_hat, mu_hat)
        ret_val = (4 / (3 * self.CTAUR)) * np.tanh(rho_value)
        ret_val -= pi_hat / tau_R
        ret_val -= (4 / 3) * pi_hat ** 2 * np.tanh(rho_value)
        return ret_val

    # Computed .
    def denergy_drho(self, ys: np.ndarray, rho_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the derivative of the energy density with respect to ρ.
        Based on the scaling relations and the derivatives of temperature and chemical potential.
        """
        # Unpack the state vector.
        T_hat, muB_hat, muS_hat, muQ_hat, _ = ys
        mu_hat = np.array([muB_hat, muS_hat, muQ_hat])
        # Compute a scaling quantity from T and chemical potentials.
        temp_value = (T_hat / self.T_ast)**2 + np.sum((mu_hat / self.mu_ast)**2)
        # Compute the derivatives dT/drho and dμ/drho:
        dT = self.dT_drho(ys, rho_val)
        dmu = self.dmu_drho(ys, rho_val)
        # Combine contributions. Here, we weight the temperature derivative and the chemical potential derivatives.
        value1 = T_hat * dT / (self.T_ast**2)
        value2 = np.sum(mu_hat * (dmu / (self.mu_ast**2)))
        return 12 * ALPHA * self.T_ast**4 * temp_value * (value1 + value2)