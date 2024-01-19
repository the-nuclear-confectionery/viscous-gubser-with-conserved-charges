from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import newton

from numpy import linspace
from numpy import ndarray
from numpy import array
from numpy import zeros
from numpy import concatenate
from numpy import sqrt
from numpy import ones_like
from numpy import meshgrid
from numpy import tanh

import matplotlib.pyplot as plt

from matplotlib.cm import copper
from matplotlib.cm import plasma
from matplotlib.cm import plasma
from matplotlib.cm import coolwarm
from matplotlib.cm import get_cmap
from matplotlib.cm import ScalarMappable

from matplotlib.patches import FancyArrow

from matplotlib.colors import Colormap
from matplotlib.colors import Normalize

from my_plotting import costumize_axis

from equations_of_motion import eom
from equations_of_motion import energy
from equations_of_motion import denergy_drho
from equations_of_motion import entropy
from equations_of_motion import number

from variable_conversions import rho
from variable_conversions import HBARC
from variable_conversions import milne_T
from variable_conversions import milne_mu

from typing import List


def find_freezeout_tau(
        e_interp: interp1d,
        e_freezeout: float,
        r: float,
        q: float,
) -> float:
    return newton(
        lambda tau: e_freezeout - e_interp(rho(tau, r, q)) / tau ** 4,
        x0=0.1,
        x1=0.2,
    )


def find_isentropic_temperature(
    mu: float,
    s_n: float,
) -> float:
    return newton(
        lambda t: 4.0 / tanh(mu / t) - mu / t - s_n,
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
        2 * q ** 2 * (r ** 2 + tau ** 2)
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


def milne_entropy(
        taus: ndarray,
        xs: ndarray,
        q: float,
        t_interp: interp1d,
        mu_interp: interp1d,
) -> ndarray:
    ts = milne_T(taus, xs, 1, t_interp)
    mus = milne_mu(taus, xs, 1, mu_interp)

    return entropy(temperature=ts, chem_potential=mus)


def solve_and_plot(
        fig: plt.Figure,
        ax: plt.Axes,
        y0s: ndarray,
        rhos_1: ndarray,
        rhos_2: ndarray,
        xs: ndarray,
        e_freezeout: float,
        q: float,
        colors: List[str],
        add_labels: bool,
        norm_scale: float = 0.1,
        heat_map: Colormap = None,
        update_color_bar: bool = False,
        plot_s_n: bool = False
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
    n_interp = interp1d(rhos, number(t_hat, mu_hat))
    s_interp = interp1d(rhos, entropy(t_hat, mu_hat))

    rho_0 = rho(tau=1.2, r=0, q=1)
    s_n = s_interp(rho_0) / n_interp(rho_0)

    skip_size = 8
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

    # some magic to make colorbars happen
    freezeout_s = milne_entropy(
        taus=freezeout_times[:, 1],
        xs=freezeout_times[:, 0],
        q=1.0,
        t_interp=t_interp,
        mu_interp=mu_interp,
    )
    
    if heat_map is None:
        heat_map = get_cmap(copper, freezeout_s.size)

        s = ax[0].scatter(
            freezeout_times[:, 0],
            freezeout_times[:, 1],
            c=freezeout_s,
            s=1.0,
            cmap=heat_map,
        )
    else:
        s = ax[0].scatter(
            freezeout_times[:, 0],
            freezeout_times[:, 1],
            c=ones_like(freezeout_s),
            s=1.0,
            cmap=heat_map.reversed(),
        )
        
    
    if update_color_bar:
        cax = fig.colorbar(s, ax=ax[0], orientation='vertical', pad=0.01,
                           format='%.2f').ax
        cax.yaxis.set_ticks(linspace(min(freezeout_s), max(freezeout_s), 7))
        for t in cax.get_yticklabels():
            t.set_fontsize(18)
        cax.set_ylabel(r'$s(\tau, x)$ [GeV$^{3}$]', fontsize=20)


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

        norm = sqrt(abs(
            normal_vectors[i, 0] ** 2 - normal_vectors[i, 1] ** 2
        ))
        normal_vectors[i] = norm_scale * normal_vectors[i] / norm

        arrows[i] = ax[0].arrow(
            x=x,
            y=tau_FO,
            dx=normal_vectors[i, 0],
            dy=normal_vectors[i, 1],
            head_width=0.01,
            head_length=0.01,
            color=colors[1]
        )

    if update_color_bar:
        x_FOs = freezeout_times[:, 0]
        tau_FOs = freezeout_times[:, 1]

        # tau0 = 1.2
        # tauf = 2.2
        # taus = linspace(tau0, tauf, 100)
        tau0 = min(tau_FOs)
        tauf = max(tau_FOs)
        taus = tau_FOs
        cmap = get_cmap(plasma, taus.size)
        norm = Normalize(vmin=tau0, vmax=tauf)
        sm = ScalarMappable(norm=norm, cmap=cmap)

        if True:
        # for xi  in linspace(1.0, 3.0, 10):
        #     ys = [y0s[0], xi * y0s[0], y0s[2]]
        #     soln_1 = odeint(eom, ys, rhos_1)
        #     soln_2 = odeint(eom, ys, rhos_2)
            t_hat = concatenate((soln_1[:, 0][::-1], soln_2[:, 0]))
            mu_hat = concatenate((soln_1[:, 1][::-1], soln_2[:, 1]))
            t_interp = interp1d(rhos, t_hat)
            mu_interp = interp1d(rhos, mu_hat)
            ax[1].scatter(
                # milne_mu(taus, 0.0, 1.0, mu_interp), 
                # milne_T(taus, 0.0, 1.0, t_interp),
                milne_mu(taus, x_FOs, 1.0, mu_interp), 
                milne_T(taus, x_FOs, 1.0, t_interp),
                c=taus,
                s=2.0,
                cmap=cmap
            )

            mus = milne_mu(taus, x_FOs, 1.0, mu_interp)
            ts = array([
                find_isentropic_temperature(
                    mu=mu,
                    s_n=s_n
                )
                for mu in mus
            ])
            ax[1].scatter(
                mus,
                ts,
                c=taus,
                s=2.0,
                cmap=cmap
            )
                    
        cax = fig.colorbar(sm, ax=ax[1], orientation='vertical', pad=0.01,
                           format='%.2f').ax
        cax.yaxis.set_ticks(linspace(tau0, tauf, 7))
        for t in cax.get_yticklabels():
            t.set_fontsize(18)
        cax.set_ylabel(r'$\tau$ [fm/c]', fontsize=20)


    if add_labels:
        xs = concatenate((-xs[::-1], xs))
        linestyles = ['solid', 'dashed', 'dotted']
        colors = ['black', 'red', 'blue']
        for n, tau in enumerate([1.2, 2.0, 3.0]):
            rh = rho(tau, xs, 1)
            ax[2].plot(
                xs,
                s_interp(rh) / n_interp(rh),
                ls=linestyles[n],
                color=colors[n],
                label=r'$\tau='+f'{tau:.2f}$ [fm/c]'
                            if add_labels else None
            )

    return heat_map


# Plot the tau components and r components separately, to see which has the large change


if __name__ == "__main__":
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(3.0 * 7, 1 * 7))
    fig.patch.set_facecolor('white')

    y0s = array([1.2, 2.4, 0.0])
    rhos_1 = linspace(-10, 0, 1000)[::-1]
    rhos_2 = linspace(0, 10, 1000)
    xs = linspace(0, 6, 1000)


    heat_map = solve_and_plot(
        fig=fig,
        ax=ax,
        y0s=y0s,
        rhos_1=rhos_1,
        rhos_2=rhos_2,
        xs=xs,
        e_freezeout=1.0 / HBARC,
        q=1,
        colors=['red', 'black'],
        update_color_bar=True,
        add_labels=True,
    )

    y0s = array([1.2, 1e-20, 0])
    rhos_1 = linspace(-30, 0, 1000)[::-1]
    rhos_2 = linspace(0, 30, 1000)
    xs = linspace(0, 6, 1000)

    solve_and_plot(
        fig=fig,
        ax=ax,
        y0s=y0s,
        rhos_1=rhos_1,
        rhos_2=rhos_2,
        xs=xs,
        e_freezeout=1.0 / HBARC,
        q=1,
        colors=['blue', 'black'],
        heat_map=heat_map,
        add_labels=False,
    )

    costumize_axis(
        ax=ax[0],
        x_title=r'$r$ [fm]',
        y_title=r'$\tau_\mathrm{FO}$ [fm/c]',
    )
    ax[0].set_xlim(0, 6.0)
    ax[0].set_ylim(0.0, 2.5)

    costumize_axis(
        ax=ax[1],
        x_title=r'$\mu(\tau, 0)$ [GeV]',
        y_title=r'$T(\tau, 0)$ [GeV]'
    )

    # ax[2].set_aspect(1.0, anchor='SW')
    costumize_axis(
        ax=ax[2],
        x_title=r'$x$ [fm]',
        y_title=r'$s(\tau, x)/n(\tau, x)$'
    )
    ax[2].legend(loc='upper right', fontsize=20)

    fig.tight_layout()
    fig.savefig('./freeze-out-surface.pdf')