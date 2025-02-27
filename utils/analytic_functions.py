import numpy as np
from scipy.interpolate import interp1d

from typing import Union
from typing import List

def rho(
        tau: float,
        r: Union[float, np.ndarray],
        q: float,
) -> float:
    return np.arcsinh(-(1 - (q * tau) ** 2 + (q * r) ** 2) / (2 * q * tau))


def kappa(
        tau: float,
        r: Union[float, np.ndarray],
        q: float,
) -> float:
    return np.arctanh((2 * q ** 2 * r * tau) /
                   (1 + (q * tau) ** 2 + (q * r) ** 2))


def u_r(
        tau: float,
        r: Union[float, np.ndarray],
        q: float,
) -> float:
    return np.sinh(kappa(tau, r, q))


def u_x(
        tau: float,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        q: float,
) -> float:
    r = np.sqrt(x ** 2 + y ** 2)
    return (x / r) * np.sinh(kappa(tau, r, q))


def u_y(
        tau: float,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        q: float,
) -> float:
    r = np.sqrt(x ** 2 + y ** 2)
    return (y / r) * np.sinh(kappa(tau, r, q))


def milne_T(
        tau: float,
        r: Union[float, np.ndarray],
        q: float,
        ads_T: interp1d,
) -> float:
    return HBARC * ads_T(rho(tau, r, q)) / tau


def milne_mu(
        tau: float,
        r: Union[float, np.ndarray],
        q: float,
        ads_mu: interp1d
) -> float:
    return HBARC * ads_mu(rho(tau, r, q)) / tau