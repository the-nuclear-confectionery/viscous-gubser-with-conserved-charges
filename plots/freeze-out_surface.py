from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import newton

from numpy import linspace
from numpy import ndarray
from numpy import array
from numpy import zeros
from numpy import zeros_like
from numpy import concatenate
from numpy import sqrt
from numpy import set_printoptions

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

from my_plotting import costumize_axis

from equations_of_motion import eom
from equations_of_motion import energy
from equations_of_motion import denergy_drho

from variable_conversions import rho
from variable_conversions import HBARC

from typing import List


def find_freezeout_tau(
        e_interp: interp1d,
        e_freezeout: float,
        x: float,
        q: float,
) -> float:
    return newton(
        lambda tau: e_freezeout - e_interp(rho(tau, x, q)) / tau ** 4,
        0.1
    )


def denergy_dtau(
        ys: ndarray,
        tau: float,
        r:float,
        q: float,
) -> float:
    temperature, chem_potenial, _ = ys
    derivative = 1 + q ** 2 * (r ** 2 + tau ** 2)
    derivative /= tau * sqrt(
        1 
        +
        q ** 4 * (r ** 2 - tau ** 2) ** 2
        +
        2 * q ** 2 * (r ** 2 + tau ** 2) ** 2
    )
    return_value = derivative * denergy_drho(ys, rho(tau, r, q)) * tau 
    return_value -= 4.0 * energy(temperature=temperature, chem_potential=chem_potenial)
    return return_value / tau ** 5


def denergy_dr(
        ys: ndarray,
        tau: float,
        r:float,
        q: float,
) -> float:
    derivative = - q * r / tau
    derivative /= sqrt(
        1 + ((1 + (q * r) ** 2 - (q * tau) ** 2 ) / (2 * q * tau)) ** 2
    )
    return derivative * denergy_drho(ys, rho(tau, r, q)) / tau ** 4


def solve_and_plot(
        ax: plt.Axes,
        y0s: ndarray,
        rhos_1: ndarray,
        rhos_2: ndarray,
        xs: ndarray,
        e_freezeout: float,
        q: float,
) -> None:
    soln_1 = odeint(eom, y0s, rhos_1)
    soln_2 = odeint(eom, y0s, rhos_2)
    t_hat = concatenate((soln_1[:, 0][::-1], soln_2[:, 0]))
    mu_hat = concatenate((soln_1[:, 1][::-1], soln_2[:, 1]))
    pi_hat = concatenate((soln_1[:, 2][::-1], soln_2[:, 2]))
    rhos = concatenate((rhos_1[::-1], rhos_2))

    t_interp = interp1d(rhos, t_hat)
    mu_interp = interp1d(rhos, mu_hat)
    pi_interp = interp1d(rhos, pi_hat)

    e_interp = interp1d(rhos, energy(t_hat, mu_hat))

    skip_size = 10 
    freezeout_times = zeros((xs.size, 2))
    normal_vectors = zeros((xs.size // skip_size, 2))
    arrows = zeros((xs.size // skip_size,), dtype=FancyArrow)

    for i, x in enumerate(xs):
        freezeout_times[i] = [
            x, 
            find_freezeout_tau(
                e_interp, e_freezeout, x, q
            )
        ]

    ax.scatter(
        freezeout_times[:, 0],
        freezeout_times[:, 1],
        color='red',
        s=1.0
    )

    for i, var in enumerate(freezeout_times[::skip_size]):
        x, tau_FO = var
        _rho = rho(tau=tau_FO, r=x, q=q)
        normal_vectors[i] = [
            -denergy_dr(
                ys=array([
                    t_interp(_rho),
                    mu_interp(_rho),
                    pi_interp(_rho)
                ]),
                tau=tau_FO,
                r=x,
                q=q
            ),
            -denergy_dtau(
                ys=array([
                    t_interp(_rho),
                    mu_interp(_rho),
                    pi_interp(_rho)
                ]),
                tau=tau_FO,
                r=x,
                q=q
            ),
        ]

        norm = abs(
            normal_vectors[i, 0] ** 2 - normal_vectors[i, 1] ** 2
        )
        normal_vectors[i] = normal_vectors[i] / norm

        arrows[i] = ax.arrow(
            x=x,
            y=tau_FO,
            dx=normal_vectors[i, 0],
            dy=normal_vectors[i, 1]
        )



if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('white')

    y0s = array([1.2, 1.2, 0])
    rhos_1 = linspace(-30, 0, 1000)[::-1]
    rhos_2 = linspace(0, 30, 1000)
    xs = linspace(0, 6, 1000)

    solve_and_plot(
        ax=ax,
        y0s=y0s,
        rhos_1=rhos_1,
        rhos_2=rhos_2,
        xs=xs,
        e_freezeout=1.0 / HBARC,
        q=1,
    )

    costumize_axis(
        ax=ax,
        x_title=r'$r$ [fm]',
        y_title=r'$\tau_\mathrm{FO}$ [fm/c]',
    )
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 3.0)
    fig.tight_layout()
    fig.savefig('./freeze-out-surface.pdf')