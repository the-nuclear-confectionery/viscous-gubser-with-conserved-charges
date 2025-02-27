from typing import Union
from numpy import pi, ndarray, array, fabs
from numpy import sum as npsum
from eos.base_eos import BaseEoS

# Degeneracy factor in the QGP
Nc = 3.0
Nf = 2.5
gQCD = 2 * (Nc ** 2 - 1) + 7 * Nc * Nf / 2
sigma = pi ** 2 / 90
gq = 4 * Nc * Nf
ALPHA = gQCD * sigma
BETA = gq
BETA1 = BETA / 216
BETA2 = BETA / (324 * pi ** 2)

class MasslessQGPEoS(BaseEoS):
    """
    Implements the massless quark-gluon plasma equation of state (EOS1 in the paper)
    with its associated equations.
    """
    def pressure(
        self,
        temperature: Union[float, ndarray],
        chem_potential: Union[float, ndarray]
    ) -> Union[float, ndarray]:
        # Note: Assumes chem_potential is a single number (baryon chemical potential)
        return_value = ALPHA * temperature ** 4
        return_value += (BETA1 * chem_potential ** 2 * temperature ** 2)
        return_value += (BETA2 * chem_potential ** 4)
        return return_value


    def energy(
        self,
        temperature: Union[float, ndarray],
        chem_potential: Union[float, ndarray]
    ) -> Union[float, ndarray]:
        # For a conformal system, e = 3p
        return 3 * self.pressure(temperature, chem_potential)

    def entropy(
        self,
        temperature: Union[float, ndarray],
        chem_potential: Union[float, ndarray]
    ) -> Union[float, ndarray]:
        return_value = (4 * ALPHA * temperature ** 3) + (2 * BETA1 * chem_potential ** 2 * temperature)
        return return_value

    def number(
        self,
        temperature: Union[float, ndarray],
        chem_potential: Union[float, ndarray]
    ) -> Union[float, ndarray]:
        return_value = (2 * BETA1 * chem_potential * temperature ** 2) + (4 * BETA2 * chem_potential ** 3)
        return return_value