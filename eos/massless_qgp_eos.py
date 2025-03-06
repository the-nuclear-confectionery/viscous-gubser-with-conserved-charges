from typing import Union
import numpy as np
from eos.base_eos import BaseEoS

# Degeneracy factor in the QGP
Nc = 3.0
Nf = 2.5
gQCD = 2 * (Nc ** 2 - 1) + 7 * Nc * Nf / 2
sigma = np.pi ** 2 / 90
gq = 4 * Nc * Nf
ALPHA = gQCD * sigma
BETA = gq
BETA1 = BETA / 216
BETA2 = BETA / (324 * np.pi ** 2)

class MasslessQGPEoS(BaseEoS):
    """
    Implements the massless quark-gluon plasma equation of state (EOS1 in the paper)
    with its associated equations.
    """

    # Eq. (12) in the paper
    def pressure(
        self,
        T: Union[float, np.ndarray],
        mu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        # Note: mu = (muB, muS, muQ), but we only use muB.
        return_value = ALPHA * T ** 4
        return_value += (BETA1 * mu[0] ** 2 * T ** 2)
        return_value += (BETA2 * mu[0] ** 4)
        return return_value

    # Conformal systems have e = 3p
    def energy(
        self,
        T: Union[float, np.ndarray],
        mu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return 3 * self.pressure(T, mu)

    def number(
        self,
        T: Union[float, np.ndarray],
        mu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        # Note: mu = (muB, muS, muQ), but we only use muB.
        # nS and nQ are always zero.
        return_value = (2 * BETA1 * mu * T ** 2) + (4 * BETA2 * mu ** 3)
        return return_value
    
    # Eq. (14) in the paper
    def entropy(
        self,
        T: Union[float, np.ndarray],
        mu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        # Note: mu = (muB, muS, muQ), but we only use muB.
        return_value = (4 * ALPHA * T ** 3) + (2 * BETA1 * mu[0] ** 2 * T)
        return return_value