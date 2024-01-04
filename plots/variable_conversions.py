from numpy import ndarray
from numpy import sinh
from numpy import arcsinh
from numpy import arctanh
from numpy import sqrt

from scipy.interpolate import interp1d

from typing import Union

from equations_of_motion import energy
from equations_of_motion import pressure

HBARC = 0.19733

def rho(
        tau: float,
        r: Union[float, ndarray],
        q: float,
) -> float:
    return arcsinh(-(1 - (q * tau) ** 2 + (q * r) ** 2) / (2 * q * tau))


def kappa(
        tau: float,
        r: Union[float, ndarray],
        q: float,
) -> float:
    return arctanh((2 * q ** 2 * r * tau) / (1 + (q * tau) ** 2 + (q * r) ** 2))


def u_r(
        tau: float,
        r: Union[float, ndarray],
        q: float,
) -> float:
    return sinh(kappa(tau, r, q))


def u_x(
        tau: float,
        x: Union[float, ndarray],
        y: Union[float, ndarray],
        q: float,
) -> float:
    r = sqrt(x ** 2 + y ** 2)
    return (x / r) * sinh(kappa(tau, r, q))


def u_y(
        tau: float,
        x: Union[float, ndarray],
        y: Union[float, ndarray],
        q: float,
) -> float:
    r = sqrt(x ** 2 + y ** 2)
    return (y / r) * sinh(kappa(tau, r, q))

def milne_T(
        tau: float,
        r: Union[float, ndarray],
        q: float,
        ads_T: interp1d,
) -> float:
    return HBARC * ads_T(rho(tau, r, q)) / tau


def milne_mu(
        tau: float,
        r: Union[float, ndarray],
        q: float,
        ads_mu: interp1d
) -> float:
    return HBARC * ads_mu(rho(tau, r, q)) / tau


def milne_pi(
        tau: float,
        x: Union[float, ndarray],
        y: Union[float, ndarray],
        q: float,
        ads_T: interp1d,
        ads_mu: interp1d,
        ads_pi_bar_hat: interp1d,
) -> float:
    r = sqrt(x ** 2 + y ** 2)
    t = ads_T(rho(tau, r, q))
    mu = ads_mu(rho(tau, r, q))
    e = energy(temperature=t, chem_potential=mu)
    p = pressure(temperature=t, chem_potential=mu)

    pi_hat = HBARC * (e + p) * ads_pi_bar_hat(rho(tau, r, q)) 
    pi_nn = pi_hat / tau ** 6
    pi_xx = -0.5 * (1 + u_x(tau, x, 0, q) ** 2) * pi_hat / tau ** 4
    pi_yy = -0.5 * (1 + u_y(tau, x, 0, q) ** 2) * pi_hat / tau ** 4
    pi_xy = -0.5 * u_x(tau, x, x, q) * u_y(tau, x, x, q) * pi_hat / tau ** 4

    return [pi_xx, pi_yy, pi_xy, pi_nn]
    