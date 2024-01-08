from numpy import cosh
from numpy import sinh
from numpy import tanh
from numpy import ndarray
from numpy import pi
from numpy import array

from typing import Union

# Note on the the units: everything is expected to be in units fm

ALPHA = 3 * pi ** 2 * (16 + 105 / 4) / 90
CTAUR = 5
ETA_S = 0.2
AA = 1.0 / 3.0

# the equations of state stuff
def pressure(
        temperature: float,
        chem_potential: float,
) -> float:
    return_value = AA * temperature ** 4 * cosh(chem_potential / temperature)
    return ALPHA * return_value


def energy(
        temperature: float,
        chem_potential: float,
) -> float:
    return_value = temperature ** 4 * cosh(chem_potential / temperature)
    return ALPHA * return_value


def entropy(
        temperature: float,
        chem_potential: float,
) -> float:
    return_value = AA * temperature ** 2 * (
        4 * temperature * cosh(chem_potential / temperature)
        -
        chem_potential * sinh(chem_potential / temperature)
    )
    return ALPHA * return_value


def number(
        temperature: float,
        chem_potential: float,
) -> float:
    return_value = AA * temperature ** 3 * sinh(chem_potential / temperature)
    return ALPHA * return_value


def tau_R(
        temperature: float,
        chem_potential: float,
) -> float:
    e = energy(temperature=temperature, chem_potential=chem_potential)
    p = pressure(temperature=temperature, chem_potential=chem_potential)
    s = entropy(temperature=temperature, chem_potential=chem_potential)
    return CTAUR * ETA_S * s / (e + p)


# The equations of motion
def dT_drho(
        ys: ndarray,
        rho: Union[float, ndarray],
) -> Union[float, ndarray]:
    
    temperature, chem_potenial, pi_hat = ys
    return_value = 2 * pi_hat 
    return_value /= 3 / cosh(chem_potenial / temperature) ** 2 + 1
    return_value += -1
    return (2 / 3) * temperature * return_value * tanh(rho)


def dmu_drho(
        ys: ndarray,
        rho: Union[float, ndarray],
) -> Union[float, ndarray]:
    temperature, chem_potenial, pi_hat = ys
    return_value = -(
        2 
        -
        6 * temperature / chem_potenial * tanh(chem_potenial / temperature)
    ) * pi_hat
    return_value += 3 / cosh(chem_potenial / temperature) ** 2
    return_value += 1
    return_value /= 3 * tanh(chem_potenial / temperature) ** 2 - 4
    return (2 / 3) * chem_potenial * return_value * tanh(rho)


def dpi_drho(
        ys: ndarray,
        rho: Union[float, ndarray]
) -> Union[float, ndarray]:
    temperature, chem_potenial, pi_hat = ys
    tau_r = tau_R(temperature=temperature, chem_potential=chem_potenial)
    return_value = (4 / 3 / CTAUR) * tanh(rho)
    return_value -= pi_hat / tau_r
    return_value -= (4 / 3) * pi_hat ** 2 * tanh(rho)
    return return_value
    

def eom(
        ys: ndarray,
        rho: Union[float, ndarray],
) -> ndarray:
    dTdrho = dT_drho(ys, rho)
    dmudrho = dmu_drho(ys, rho)
    dpidrho = dpi_drho(ys, rho)

    return array([dTdrho, dmudrho, dpidrho])


def denergy_drho(
        ys: ndarray,
        rho: Union[float, ndarray],
) -> ndarray:
    temperature, chem_potenial, _ = ys
    return_value_1 = 4 * cosh(chem_potenial / temperature) * temperature 
    return_value_1 *= dT_drho(ys, rho)
    return_value_2 = -chem_potenial * dT_drho(ys, rho) 
    return_value_2 += temperature * dmu_drho(ys, rho)
    return_value_2 *= sinh(chem_potenial / temperature)
    return ALPHA * temperature ** 2 * (return_value_1 + return_value_2)


# equations of motion for alternative EoS
def dT_drho_alt(
        ys: ndarray,
        rho: Union[float, ndarray],
) -> Union[float, ndarray]:
    
    temperature, chem_potenial, pi_hat = ys
    return_value = (-2 + pi_hat) / 3.0
    return_value += pi_hat * (chem_potenial / pi / temperature) ** 2
    return temperature * return_value * tanh(rho)


def dmu_drho_alt(
        ys: ndarray,
        rho: Union[float, ndarray],
) -> Union[float, ndarray]:
    _, chem_potenial, pi_hat = ys
    return_value = 1 + pi_hat
    return -(2 / 3) * chem_potenial * return_value * tanh(rho)


def dpi_drho_alt(
        ys: ndarray,
        rho: Union[float, ndarray]
) -> Union[float, ndarray]:
    temperature, chem_potenial, pi_hat = ys
    tau_r = tau_R(temperature=temperature, chem_potential=chem_potenial)
    return_value = (4 / 3 / CTAUR) * tanh(rho)
    return_value -= pi_hat / tau_r
    return_value -= (4 / 3) * pi_hat ** 2 * tanh(rho)
    return return_value
    

def eom_alt(
        ys: ndarray,
        rho: Union[float, ndarray],
) -> ndarray:
    dTdrho = dT_drho_alt(ys, rho)
    dmudrho = dmu_drho_alt(ys, rho)
    dpidrho = dpi_drho_alt(ys, rho)

    return array([dTdrho, dmudrho,dpidrho])