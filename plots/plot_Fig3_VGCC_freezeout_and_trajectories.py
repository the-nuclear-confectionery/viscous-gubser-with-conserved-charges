import sys
sys.path.append('..')

from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import newton

from numpy import *

import matplotlib.pyplot as plt

from matplotlib.cm import copper
from matplotlib.cm import plasma
from matplotlib.cm import viridis
from matplotlib.cm import get_cmap
from matplotlib.cm import ScalarMappable

from matplotlib.patches import FancyArrow

from matplotlib.colors import Colormap
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm

from plots.plotting_settings import *


from eos.system_massless_qgp import MasslessQGP
from eos.variable_conversions import HBARC
from eos.variable_conversions import rho
from eos.variable_conversions import milne_T
from eos.variable_conversions import milne_mu

from typing import List
from typing import Union
from typing import Tuple

system = MasslessQGP()


def find_freezeout_tau(
        e_interp: interp1d,
        e_freezeout: float,
        r: float,
        q: float,
) -> float:
    def f(tau: float) -> float:
        return e_freezeout - e_interp(rho(tau, r, q)) / tau ** 4

    try:
        value = newton(
            f,
            x0=0.01,
            x1=0.02,
        )
    except (ValueError, RuntimeError):
        value = newton(
            f,
            x0=0.001,
            x1=0.002,
        )
    return value


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
        r: float,
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
    return_value = derivative * system.denergy_drho(ys, rho(tau, r, q)) * tau
    return_value -= 4.0 * \
        system.eos.energy(temperature=temperature, chem_potential=chem_potenial)
    return return_value / tau ** 5


def denergy_dr(
        ys: ndarray,
        tau: float,
        r: float,
        q: float,
) -> float:
    derivative = - q * r / tau
    derivative /= sqrt(
        1 + ((1 + (q * r) ** 2 - (q * tau) ** 2) / (2 * q * tau)) ** 2
    )
    return derivative * system.denergy_drho(ys, rho(tau, r, q)) / tau ** 4


def find_color_indices(
        max_s: float,
        min_s: float,
        num_colors: int,
        s: Union[float, ndarray],
) -> Union[float, ndarray]:
    ds = (max_s - min_s) / num_colors
    return (s - min_s) // ds


def do_freezeout_surfaces(
        rhos_1: ndarray,
        rhos_2: ndarray,
        xs: ndarray,
        T0: float,
        mu0_T0_ratios: ndarray,
        pi0: float,
        e_freezeout: float,
        skip_size: int,
        norm_scale: float,
        q: float
) -> Tuple[
        float,
        float,
        float,
        float,
        List[ndarray],
        List[ndarray],
        List[ndarray]
]:
    # This function does a lot more work than it needs to...
    # returns:
    #   - min entropy at freezeout for all ICs
    #   - max entropy at freezeout for al ICs
    #   - min freeze-out time for all ICs
    #   - max freeze-out time for all ICs
    #   - list of freeze-out surface (r_FO, tau_FO) arrays for ICs
    #   - list of entropy densities of freeze-out surface for all ICs
    #   - list of normalized normal vectors for freeze-out surfaces

    list_FO_surfaces = []
    list_FO_entropies = []
    list_FO_normals = []
    min_s = 1e99
    max_s = -1e99
    min_tau = 1e99
    max_tau = -1e99

    for k, alpha in enumerate(mu0_T0_ratios):
        y0s = [T0, alpha * T0, pi0]
        soln_1 = odeint(system.eom.for_scipy, y0s, rhos_1)
        soln_2 = odeint(system.eom.for_scipy, y0s, rhos_2)
        t_hat = concatenate((soln_1[:, 0][::-1], soln_2[:, 0]))
        mu_hat = concatenate((soln_1[:, 1][::-1], soln_2[:, 1]))
        pi_hat = concatenate((soln_1[:, 2][::-1], soln_2[:, 2]))
        rhos = concatenate((rhos_1[::-1], rhos_2))

        t_interp = interp1d(rhos, t_hat)
        mu_interp = interp1d(rhos, mu_hat)
        pi_interp = interp1d(rhos, pi_hat)

        e_interp = interp1d(rhos, system.eos.energy(t_hat, mu_hat))

        freezeout_times = zeros((xs.size, 2))

        for i, x in enumerate(xs):
            freezeout_times[i] = [
                x,
                find_freezeout_tau(
                    e_interp, e_freezeout, x, q
                )
            ]

        list_FO_surfaces.append(freezeout_times)
        min_tau = min(fmin(freezeout_times[:, 1], min_tau))
        max_tau = max(fmax(freezeout_times[:, 1], max_tau))

        # some magic to make colorbars happen
        freezeout_s = system.milne_entropy(
            tau=freezeout_times[:, 1],
            x=freezeout_times[:, 0],
            y=0.0,
            q=1.0,
            ads_T=t_interp,
            ads_mu=mu_interp,
        )

        list_FO_entropies.append(freezeout_s)
        min_s = min(fmin(freezeout_s, min_s))
        max_s = max(fmax(freezeout_s, max_s))

        normal_vectors = zeros((xs.size // skip_size, 2))
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
        list_FO_normals.append(normal_vectors)

    return min_s, max_s, min_tau, max_tau, list_FO_surfaces, \
        list_FO_entropies, list_FO_normals


def dx_dtau(
        ys: ndarray,
        tau: Union[float, ndarray],
        q: float
) -> Union[float, ndarray]:
    x, y, eta = ys
    r2 = x ** 2 + y ** 2
    v_r = 2 * q * tau / (1 + q ** 2 * (r2 + tau ** 2))
    r = sqrt(r2)
    return array([x / r, y / y, 0]) * v_r


def solve_and_plot(
        fig: plt.Figure,
        ax: plt.Axes,
        y0s: ndarray,
        mu0_T0_ratios: ndarray,
        rhos_1: ndarray,
        rhos_2: ndarray,
        xs: ndarray,
        e_freezeout: float,
        q: float,
        add_labels: bool = True,
        norm_scale: float = 0.1,
        heat_map: Colormap = None,
        update_color_bar: bool = False,
        plot_s_n: bool = False,
) -> None:

    skip_size = 8
    min_s, max_s, min_tau, max_tau, fo_surfaces, \
        fo_entropies, fo_normals = do_freezeout_surfaces(
            rhos_1=rhos_1,
            rhos_2=rhos_2,
            xs=xs,
            T0=y0s[0],
            mu0_T0_ratios=mu0_T0_ratios,
            pi0=y0s[2],
            skip_size=skip_size,
            e_freezeout=e_freezeout,
            norm_scale=norm_scale,
            q=q,
        )

    evol_taus_log = linspace(log(0.01), log(10), 1000)
    evol_taus = exp(evol_taus_log)

    xis = mu0_T0_ratios
    for itr in range(mu0_T0_ratios.size):
        freezeout_times = fo_surfaces[itr]
        freezeout_s = fo_entropies[itr]
        normal_vectors = fo_normals[itr]

        arrows = zeros((xs.size // skip_size,), dtype=FancyArrow)
        for i, var in enumerate(freezeout_times[::skip_size]):
            x, tau_FO = var
            arrows[i] = ax[0].arrow(
                x=x,
                y=tau_FO,
                dx=normal_vectors[i, 0],
                dy=normal_vectors[i, 1],
                head_width=0.02,
                head_length=0.01,
                lw=0.5,
                color='black'
            )

        heat_map = get_cmap('copper', freezeout_s.size)

        ax[0].scatter(
            freezeout_times[:, 0],
            freezeout_times[:, 1],
            c=find_color_indices(
                min_s=min_s,
                max_s=max_s,
                num_colors=freezeout_s.size,
                s=freezeout_s,
            ),
            s=0.5,
            cmap=heat_map,
            norm=Normalize(vmin=0, vmax=freezeout_s.size)
        )

        if update_color_bar and itr == 1:
            norm = Normalize(
                vmin=min_s,
                vmax=max_s,
            )
            s = ScalarMappable(
                norm=norm,
                cmap=heat_map
            )
            cax = fig.colorbar(s, ax=ax[0], orientation='vertical', pad=0.02,
                               format='%.2f').ax
            cax.yaxis.set_ticks(linspace(min_s, max_s, 7))
            for t in cax.get_yticklabels():
                t.set_fontsize(10)
            cax.set_ylabel(r'$\displaystyle s(\tau, x)$ [GeV$^{3}$]', fontsize=12)

        if itr == 0:
            continue

        xi = xis[itr]
        ys = [y0s[0], xi * y0s[0], y0s[2]]
        soln_1 = odeint(system.eom.for_scipy, ys, rhos_1)
        soln_2 = odeint(system.eom.for_scipy, ys, rhos_2)
        rhos = concatenate((rhos_1[::-1], rhos_2))
        t_hat = concatenate((soln_1[:, 0][::-1], soln_2[:, 0]))
        mu_hat = concatenate((soln_1[:, 1][::-1], soln_2[:, 1]))
        t_interp = interp1d(rhos, t_hat)
        mu_interp = interp1d(rhos, mu_hat)

        rs = linspace(0.01, 1.0, 100)
        evol_mus = zeros((rs.size, evol_taus.size))
        evol_temps = zeros_like(evol_mus)
        for nn, r0 in enumerate(rs):
            evol_xs = odeint(dx_dtau, array(
                [r0, r0, 0]), exp(evol_taus_log), args=(1.0,))
            evol_rs = sqrt(evol_xs[:, 0] ** 2 + evol_xs[:, 1] ** 2)
            evol_mus[nn] = milne_mu(evol_taus, evol_rs, 1.0, mu_interp)
            evol_temps[nn] = milne_T(evol_taus, evol_rs, 1.0, t_interp)

        _, _, _, color_mesh = ax[itr].hist2d(
            evol_mus.reshape(-1,),
            evol_temps.reshape(-1,),
            bins=100,
            cmap=[viridis, plasma][itr - 1],
            norm='log',
            alpha=1.0,
            density=True,
        )

        # Plot ideal trajectories
        xi = xis[itr]
        ys = [y0s[0], xi * y0s[0], y0s[2]]
        soln_1 = odeint(system.eom.for_scipy, ys, rhos_1, args=(True,))
        soln_2 = odeint(system.eom.for_scipy, ys, rhos_2, args=(True,))
        rhos = concatenate((rhos_1[::-1], rhos_2))
        t_hat = concatenate((soln_1[:, 0][::-1], soln_2[:, 0]))
        mu_hat = concatenate((soln_1[:, 1][::-1], soln_2[:, 1]))
        t_interp = interp1d(rhos, t_hat)
        mu_interp = interp1d(rhos, mu_hat)

        evol_mus = zeros((rs.size, evol_taus.size))
        evol_temps = zeros_like(evol_mus)
        for nn, r0 in enumerate([1e-5]):
            evol_xs = odeint(dx_dtau, array(
                [r0, r0, 0]), exp(evol_taus_log), args=(1.0,))
            evol_rs = sqrt(evol_xs[:, 0] ** 2 + evol_xs[:, 1] ** 2)
            evol_mus = milne_mu(evol_taus, evol_rs, 1.0, mu_interp)
            evol_temps = milne_T(evol_taus, evol_rs, 1.0, t_interp)

            ax[itr].plot(evol_mus, evol_temps, lw=1, color='black', ls='dashed')

        # ax[itr].set_ylim(bottom=0, top=2)
        # ax[itr].set_xlim(left=0, right=1.1)

        cax_2 = fig.colorbar(color_mesh, ax=ax[itr], orientation='vertical',
                             pad=0.02, format='%.2f').ax
        for t in cax_2.get_yticklabels():
            t.set_fontsize(10)
        cax_2.set_ylabel(r'count (normalized)', fontsize=12)

    return heat_map


def main():
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11.5, 3.36), dpi=1200)
    fig.patch.set_facecolor('white')

    y0s = array([1.2, 1 * 1.2, 0.0])
    rhos_1 = linspace(-10, 0, 1000)[::-1]
    rhos_2 = linspace(0, 10, 1000)
    xs = linspace(0, 6, 1000)

    solve_and_plot(
        fig=fig,
        ax=ax,
        y0s=y0s,
        mu0_T0_ratios=array([1e-20, 5.0, 8.0]),
        rhos_1=rhos_1,
        rhos_2=rhos_2,
        xs=xs,
        e_freezeout=1.0 / HBARC,
        q=1,
        # colors=['blue'],
        update_color_bar=True,
        plot_s_n=True,
        norm_scale=0.05
    )

    customize_axis(
        ax=ax[0],
        x_title=r'$\displaystyle r$ [fm]',
        y_title=r'$\displaystyle\tau_\mathrm{FO}$ [fm/$c$]',
    )
    ax[0].set_xlim(0, 6.0)
    ax[0].set_ylim(-0.1, 3.1)
    ax[0].text(3.6, 2.5, r'$\displaystyle \hat{\mu}_{Y,0}/\hat{T}_0=3$', fontsize=10)
    ax[0].text(0.9, 2.2, r'$\displaystyle \hat{\mu}_{Y,0}/\hat{T}_0=2$', fontsize=10)
    # ax[0].text(3.4, 0.55, r'$\displaystyle \hat{\mu}_{Y,0}/\hat{T}_0=1$', fontsize=10)
    ax[0].text(0.15, 0.5, r'$\displaystyle \hat{\mu}_{Y,0}/\hat{T}_0=0$', fontsize=10)

    customize_axis(
        ax=ax[1],
        x_title=r'$\displaystyle \mu_Y$ [GeV]',
        y_title=r'$\displaystyle T$ [GeV]',
        xlim=(0, 8),
        ylim=(-0.1, 3.1),
    )
    ax[1].axhline(0.2, color='black', lw=0.6)
    ax[1].text(4.0, 0.25, '$\displaystyle T=200$ MeV', fontsize=10)
    ax[1].text(4.2, 0.95, '$\displaystyle s/n_Y=\\; $const.', rotation=28, fontsize=10)
    ax[1].text(0.06, 0.82, r'$\displaystyle \hat{\mu}_{Y,0}/\hat{T}_0=2$', fontsize=10, transform=ax[1].transAxes)

    customize_axis(
        ax=ax[2],
        x_title=r'$\displaystyle \mu_Y$ [GeV]',
        y_title=r'$\displaystyle T$ [GeV]',
        xlim=(0, 10),
        ylim=(-0.2, 6.2),
    )
    ax[2].axhline(0.2, color='black', lw=0.6)
    ax[2].text(5.9, 0.30, '$\displaystyle T=200$ MeV', fontsize=10)
    ax[2].text(5.7, 0.85, '$\displaystyle s/n_Y=\\; $const.', rotation=12, fontsize=10)
    ax[2].text(0.06, 0.82, r'$\displaystyle \hat{\mu}_{Y,0}/\hat{T}_0=3$', fontsize=10, transform=ax[2].transAxes)
    for name, label in zip(range(3), ['a', 'b', 'c']):
        ax[name].text(
            0.215,
            0.91,
            r'\textsc{EoS1}',
            transform=ax[name].transAxes,
            fontsize=10,
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'linewidth': 0.5},
            horizontalalignment='center',
        )
        ax[name].text(
            0.07, 0.91,
            f'({label})',
            transform=ax[name].transAxes,
            fontsize=10,
            horizontalalignment='center'
        )


    # ax[1].set_yscale('log')
    # ax[1].text(0.1, 0.7, r'$\hat{\mu}_{Y,0}/\hat{T}_0=1$', fontsize=18)
    # ax[1].text(0.65, 0.7, r'$\hat{\mu}_{Y,0}/\hat{T}_0=2$', fontsize=18)
    # ax[1].text(1.05, 0.7, r'$\hat{\mu}_{Y,0}/\hat{T}_0=3$', fontsize=18)

    # ax[1].text(0.01, 0.159, r'$+0.01$', fontsize=16)
    # ax[1].text(0.01, 0.169, r'$+0.02$', fontsize=16)

    # ax[2].set_aspect(1.0, anchor='SW')
    # customize_axis(
    #     ax=ax[2],
    #     x_title=r'$x$ [fm]',
    #     y_title=r'$s(\tau, x)/n_Y(\tau, x)$'
    # )
    # ax[2].legend(loc='upper center', fontsize=20)

    fig.tight_layout()
    fig_name = './output/Fig3_VGCC-freezeout_and_trajectories.pdf'
    print(f'Saving figure to {fig_name}')
    fig.savefig(fig_name)


if __name__ == "__main__":
    main()
