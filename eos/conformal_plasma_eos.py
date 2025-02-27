import numpy as np
from typing import Union, Optional, Dict

from eos.base_eos import BaseEoS

# Degeneracy factor in the QGP
Nc = 3.0
Nf = 2.5
gQCD = 2 * (Nc ** 2 - 1) + 7 * Nc * Nf / 2
sigma = np.pi ** 2 / 90
ALPHA = gQCD * sigma

class ConformalPlasmaEoS(BaseEoS):
    """
    Implements the conformal plasma equation of state (EoS2 in the paper)
    using the scaling relations.
    
    The implementation assumes that the reference values for the temperature and
    chemical potentials are provided via the constructor.
    """
    def __init__(self, eos_params: Dict[str, Union[float, np.ndarray]]):
        T_ast = eos_params.get("T_ast")
        mu_ast = eos_params.get("mu_ast")
        if T_ast is None or mu_ast is None:
            raise ValueError("Both T_ast and mu_ast must be provided inside params for conformal_plasma.")

        self.T_ast = T_ast
        if type(mu_ast) == float:
            self.mu_ast = np.array([mu_ast, mu_ast, mu_ast])
        else:
            self.mu_ast = mu_ast

    # Eq. (13) in the paper
    def pressure(
        self,
        T: Union[float, np.ndarray],
        mu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        mu = np.array(mu)
        return_value = (T / self.T_ast) ** 2
        return_value += np.sum((mu / self.mu_ast) ** 2)
        return_value = self.T_ast ** 4 * return_value ** 2
        return ALPHA * return_value

    # Conformal systems have e = 3p
    def energy(
        self,
        T: Union[float, np.ndarray],
        mu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        mu = np.array(mu)
        return 3 * self.pressure(T, mu)

    # Eq. (14) in the paper
    def number(
        self,
        T: Union[float, np.ndarray],
        mu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        mu = np.array(mu)
        ratio = self.T_ast / self.mu_ast
        reduced_T = T / self.T_ast
        reduced_mu = mu / self.mu_ast
        return_value = 4 * ALPHA * (self.T_ast ** 3) * ratio * reduced_mu * (reduced_T ** 2 + np.sum(reduced_mu ** 2))
        return return_value

    # Eq. (14) in the paper
    def entropy(
        self,
        T: Union[float, np.ndarray],
        mu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        mu = np.array(mu)
        reduced_T = T / self.T_ast
        reduced_mu = mu / self.mu_ast
        return_value = 4 * ALPHA * (self.T_ast ** 3) * reduced_T * (reduced_T ** 2 + np.sum(reduced_mu ** 2))
        return return_value