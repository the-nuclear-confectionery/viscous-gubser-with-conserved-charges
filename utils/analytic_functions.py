import numpy as np
from scipy.interpolate import interp1d

from typing import Union

from utils.constants import HBARC, tolerance
from eos.base_eos import BaseEoS

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
        interpolated_T_hat: interp1d,
) -> float:
    return HBARC * interpolated_T_hat(rho(tau, r, q)) / tau


def milne_mu(
        tau: float,
        r: Union[float, np.ndarray],
        q: float,
        interpolated_mu_hat: interp1d
) -> float:
    return HBARC * interpolated_mu_hat(rho(tau, r, q)) / tau


def milne_energy(tau: float,
                x: Union[float, np.ndarray],
                y: Union[float, np.ndarray],
                q: float,
                interpolated_T_hat: interp1d,
                interpolated_mu_hat: list,
                eos_instance: BaseEoS) -> Union[float, np.ndarray]:
        r = np.sqrt(x**2 + y**2)
        rho_value = rho(tau, r, q)
        T_hat = interpolated_T_hat(rho_value)
        mu_hat = np.array([f(rho_value) for f in interpolated_mu_hat])
        if isinstance(T_hat, np.ndarray):
                energy_hat = eos_instance.energy(T_hat, mu_hat)
                result = HBARC * energy_hat / tau ** 4
                return result
        else:
                if T_hat <= tolerance:
                        T_hat = tolerance
                energy_hat = eos_instance.energy(T_hat, mu_hat)
                result = HBARC * energy_hat / tau ** 4
                return tolerance if result < tolerance else result


def milne_number(tau: float, 
                x: Union[float, np.ndarray], 
                y: Union[float, np.ndarray], 
                q: float, 
                interpolated_T_hat: interp1d, 
                interpolated_mu_hat: list, 
                eos_instance: BaseEoS) -> Union[float, np.ndarray]:
        r = np.sqrt(x ** 2 + y ** 2)
        rho_value = rho(tau, r, q)
        T_hat = interpolated_T_hat(rho_value)
        mu_hat = np.array([f(rho_value) for f in interpolated_mu_hat])
        if isinstance(T_hat, np.ndarray):
                number_hat = eos_instance.number(T_hat, mu_hat)
                return number_hat / tau ** 3
        else:
                if T_hat <= tolerance:
                        T_hat = tolerance
                number_hat = eos_instance.number(T_hat, mu_hat)
                n = number_hat / tau ** 3
                n[np.where(n < tolerance)] = tolerance
                return n


def milne_entropy(tau: float, 
                x: Union[float, np.ndarray], 
                y: Union[float, np.ndarray], 
                q: float, 
                interpolated_T_hat: interp1d, 
                interpolated_mu_hat: list, 
                eos_instance: BaseEoS) -> Union[float, np.ndarray]:
        r = np.sqrt(x ** 2 + y ** 2)
        rho_value = rho(tau, r, q)
        T_hat = interpolated_T_hat(rho_value)
        mu_hat = np.array([f(rho_value) for f in interpolated_mu_hat])
        if isinstance(T_hat, np.ndarray):
                entropy_hat = eos_instance.entropy(T_hat, mu_hat)
                return entropy_hat / tau ** 3
        else:
                if T_hat <= tolerance:
                        T_hat = tolerance
                entropy_hat = eos_instance.entropy(T_hat, mu_hat)
                s = entropy_hat / tau ** 3
                return tolerance if s < tolerance else s


def milne_pi(tau: float, 
                x: Union[float, np.ndarray], 
                y: Union[float, np.ndarray], 
                q: float, 
                interpolated_T_hat: interp1d, 
                interpolated_mu_hat: list,
                interpolated_pi_bar_hat: interp1d, 
                eos_instance: BaseEoS,
                nonzero_xy: bool = False) -> list:
        r = np.sqrt(x ** 2 + y **2)
        rho_value = rho(tau, r, q)
        T_hat = interpolated_T_hat(rho_value)
        mu_hat = np.array([f(rho_value) for f in interpolated_mu_hat])
        if not isinstance(T_hat, np.ndarray):
                if T_hat <= tolerance:
                        T_hat = tolerance
        e_val = eos_instance.energy(T_hat, mu_hat)
        p_val = eos_instance.pressure(T_hat, mu_hat)
        pi_hat = HBARC * (e_val + p_val) * interpolated_pi_bar_hat(rho_value)
        pi_nn = pi_hat / tau ** 6
        pi_xx = -0.5 * (1 + u_x(tau, x, y, q) ** 2) * pi_hat / tau**4
        pi_yy = -0.5 * (1 + u_y(tau, x, y, q) ** 2) * pi_hat / tau**4
        if nonzero_xy:
                y = x
        pi_xy = -0.5 * u_x(tau, x, y, q) * u_y(tau, x, y, q) * pi_hat / tau ** 4
        if not isinstance(T_hat, np.ndarray):
                pi_nn = tolerance if np.fabs(pi_nn) < tolerance else pi_nn
                pi_xx = tolerance if np.fabs(pi_xx) < tolerance else pi_xx
                pi_yy = tolerance if np.fabs(pi_yy) < tolerance else pi_yy
                pi_xy = tolerance if np.fabs(pi_xy) < tolerance else pi_xy
        return [pi_xx, pi_yy, pi_xy, pi_nn]