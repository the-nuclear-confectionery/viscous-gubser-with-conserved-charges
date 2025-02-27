#!/usr/bin/env python3
"""
Plot Fig1: VGCC evolution of Temperature, Chemical Potential, Shear components, Energy density, 
Number density, and Entropy density for a massless QGP.

This script uses the configuration in a YAML file and the EOS/EoM factories to set up the system.
It then solves the evolution equations, interpolates the Milne variables, and produces two figures.
"""

import sys
import os
import argparse
import yaml
import logging

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import List

# Append parent directory so we can import our local modules.
sys.path.append('..')

from eos.eos_factory import get_eos
from eom.eom_factory import get_eom as get_custom_eom
from plots.plotting_settings import customize_axis  # Assumed to be defined in your project
from utils.analytic_functions import milne_T, milne_mu

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Config class similar to vgcc.py
class Config:
    def __init__(self, config: dict):
        self.tau_0: float = config.get('tau_0', 1.0)
        self.tau_f: float = config.get('tau_f', 1.6)
        self.tau_step: float = config.get('tau_step', 0.10)
        self.x_min: float = config.get('x_min', -5.0)
        self.x_max: float = config.get('x_max', 5.0)
        self.x_step: float = config.get('x_step', 0.025)
        self.y_min: float = config.get('y_min', -5.0)
        self.y_max: float = config.get('y_max', 5.0)
        self.y_step: float = config.get('y_step', 0.025)
        self.temp_0: float = config.get('temp_0', 0.250)
        self.muB_0: float = config.get('muB_0', 0.050) or 1e-20
        self.muS_0: float = config.get('muS_0', 0.000) or 1e-20
        self.muQ_0: float = config.get('muQ_0', 0.000) or 1e-20
        self.ceos_temp_0: float = config.get('ceos_temp_0', 1.000)
        self.ceos_muB_0: float = config.get('ceos_muB_0', 1.00)
        self.ceos_muS_0: float = config.get('ceos_muS_0', 1.00)
        self.ceos_muQ_0: float = config.get('ceos_muQ_0', 1.00)
        self.pi_0: float = config.get('pi_0', 0.0)
        self.tol: float = config.get('tolerance', 1.0e-12)
        self.output_dir: str = config.get('output_dir', "data/output/analytic_solutions")

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return cls(config)


# Define subplot indices (for the left figure arranged in a 2x2 grid)
T_PLOT = (0, 0)      # Temperature plot
MU_PLOT = (0, 1)     # Chemical potential plot
PIYY_PLOT = (1, 0)   # Shear component (pi^yy normalized) plot
PIXY_PLOT = (1, 1)   # Shear component (pi^xy normalized) plot

# For the right figure with 1x3 grid:
E_PLOT = 0  # Energy density
N_PLOT = 1  # Number density
S_PLOT = 2  # Entropy density


def solve_and_plot(
    ax_1: np.ndarray,
    ax_2: np.ndarray,
    y0s: np.ndarray,
    rhos_1: np.ndarray,
    rhos_2: np.ndarray,
    xs: np.ndarray,
    taus: np.ndarray,
    colors: List[str],
    linestyles: List[str],
    add_labels: bool = False,
) -> None:
    """
    Solve the evolution equations and plot the Milne evolution on the provided axes.
    """
    # Solve for negative and positive rho ranges
    soln_1 = odeint(system.eom.for_scipy, y0s, rhos_1)
    soln_2 = odeint(system.eom.for_scipy, y0s, rhos_2)
    t_hat = np.concatenate((soln_1[:, 0][::-1], soln_2[:, 0]))
    mu_hat = np.concatenate((soln_1[:, 1][::-1], soln_2[:, 1]))
    pi_bar_hat = np.concatenate((soln_1[:, 2][::-1], soln_2[:, 2]))
    rhos = np.concatenate((rhos_1[::-1], rhos_2))

    # Interpolation functions for the evolution variables
    t_interp = interp1d(rhos, t_hat)
    mu_interp = interp1d(rhos, mu_hat)
    pi_interp = interp1d(rhos, pi_bar_hat)

    # Loop over the provided set of proper-times tau
    for n, tau in enumerate(taus):
        t_evol = milne_T(tau, xs, 1, t_interp)
        mu_evol = milne_mu(tau, xs, 1, mu_interp)
        e_evol = system.milne_energy(tau, xs, 0.0, 1.0, t_interp, mu_interp)
        n_evol = system.milne_number(tau, xs, 0.0, 1.0, t_interp, mu_interp)
        s_evol = system.milne_entropy(tau, xs, 0.0, 1.0, t_interp, mu_interp)

        # Plot temperature on the left grid
        ax_1[T_PLOT].plot(
            xs, t_evol,
            color=colors[n],
            lw=1,
            ls=linestyles[n],
            label=(r'$\displaystyle\hat{\mu}_{Y,\,0}/\hat{T}_0=' + f'{y0s[1] / y0s[0]:.1f}$'
                   if n == 0 else None)
        )
        # Plot chemical potential on the left grid
        ax_1[MU_PLOT].plot(
            xs, mu_evol,
            color=colors[n],
            lw=1,
            ls=linestyles[n],
            label=(r'$\displaystyle \tau = ' + f'{tau:.2f}$ [fm/$c$]' if add_labels else None)
        )
        # Compute shear components from the Milne conversion
        pi_xx, pi_yy, pi_xy, _ = system.milne_pi(
            tau, xs, 0.0, 1,
            t_interp, mu_interp, pi_interp,
            nonzero_xy=True
        )
        ax_1[PIYY_PLOT].plot(
            xs, pi_yy / (4.0 * e_evol / 3.0),
            color=colors[n],
            lw=1,
            ls=linestyles[n]
        )
        ax_1[PIXY_PLOT].plot(
            xs, pi_xy / (4.0 * e_evol / 3.0),
            color=colors[n],
            lw=1,
            ls=linestyles[n]
        )

        # Plot the right figure curves: energy, number, and entropy
        ax_2[E_PLOT].plot(
            xs, e_evol,
            color=colors[n],
            lw=1,
            ls=linestyles[n],
            label=(r'$\displaystyle\hat{\mu}_{Y,\,0}/\hat{T}_0=' + f'{y0s[1] / y0s[0]:.1f}$'
                   if n == 0 else None)
        )
        ax_2[N_PLOT].plot(
            xs, n_evol,
            color=colors[n],
            lw=1,
            ls=linestyles[n],
            label=(r'$\displaystyle \tau = ' + f'{tau:.2f}$ [fm/$c$]' if add_labels else None)
        )
        ax_2[S_PLOT].plot(
            xs, s_evol,
            color=colors[n],
            lw=1,
            ls=linestyles[n]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Fig1: VGCC evolution for T, μ, shear components, energy, number & entropy densities."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to the config YAML file")
    parser.add_argument("--eos", default="massless_qgp",
                        choices=["massless_qgp", "conformal_plasma"],
                        help="Equation of state type to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--output1", default="./output/Fig1_VGCC-T_mu_pi.pdf",
                        help="Output file for the left figure (T, μ, shear)")
    parser.add_argument("--output2", default="./output/Fig2_VGCC-e_n_s.pdf",
                        help="Output file for the right figure (e, n, s)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Load configuration
    try:
        cfg = Config.from_yaml(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Create the EOS instance using the factory.
    try:
        eos_instance = get_eos(
            args.eos,
            temperature_0=cfg.ceos_temp_0,
            chem_potential_0=np.array([cfg.ceos_muB_0, cfg.ceos_muS_0, cfg.ceos_muQ_0])
        )
    except ValueError as exc:
        logger.error(f"EoS type {args.eos} is not available: {exc}")
        sys.exit(1)

    # Create the corresponding EoM instance using the factory.
    try:
        eom_instance = get_custom_eom(
            args.eos,
            temperature_0=cfg.ceos_temp_0,
            chem_potential_0=np.array([cfg.ceos_muB_0, cfg.ceos_muS_0, cfg.ceos_muQ_0])
        )
    except ValueError as exc:
        logger.error(f"EoM type {args.eos} is not available: {exc}")
        sys.exit(1)

    # Set up initial conditions (scaled variables)
    global system  # needed for solve_and_plot; alternatively pass system as argument.
    system = eos_instance  # For compatibility with our milne conversion calls in our system.EoM methods.
    y0s = np.array([
        cfg.temp_0 * cfg.tau_0 / eos_instance.HBARC,
        cfg.muB_0 * cfg.tau_0 / eos_instance.HBARC,
        cfg.muS_0 * cfg.tau_0 / eos_instance.HBARC,
        cfg.muQ_0 * cfg.tau_0 / eos_instance.HBARC,
        cfg.pi_0
    ])

    # Define spatial grid and rho ranges.
    xs = np.linspace(-6, 6, 200)
    rhos_1 = np.linspace(-10, 0, 1000)[::-1]
    rhos_2 = np.linspace(0, 10, 1000)

    # Prepare figure 1 (left; 2x2 grid) and figure 2 (right; 1x3 grid)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 6), dpi=1200, sharex=True)
    fig.patch.set_facecolor('white')
    fig2, ax2 = plt.subplots(nrows=1, ncols=3, figsize=(10.5, 3.36), dpi=1200)

    # First set of initial conditions (low μ)
    solve_and_plot(
        ax_1=ax,
        ax_2=ax2,
        y0s=y0s,  # low μ: y0s defined above
        rhos_1=rhos_1,
        rhos_2=rhos_2,
        xs=xs,
        taus=np.array([1.2, 2.0, 3.0]),
        colors=3 * ['black'],
        linestyles=['solid', 'dashed', 'dotted'],
        add_labels=True
    )

    # Second set (moderate μ)
    y0s = np.array([cfg.temp_0, 5 * cfg.temp_0, 0.0, 0.0, cfg.pi_0])
    solve_and_plot(
        ax_1=ax,
        ax_2=ax2,
        y0s=y0s,
        rhos_1=rhos_1,
        rhos_2=rhos_2,
        xs=xs,
        taus=np.array([1.2, 2.0, 3.0]),
        colors=3 * ['red'],
        linestyles=['solid', 'dashed', 'dotted']
    )

    # Third set (high μ)
    y0s = np.array([cfg.temp_0, 8 * cfg.temp_0, 0.0, 0.0, cfg.pi_0])
    solve_and_plot(
        ax_1=ax,
        ax_2=ax2,
        y0s=y0s,
        rhos_1=rhos_1,
        rhos_2=rhos_2,
        xs=xs,
        taus=np.array([1.2, 2.0, 3.0]),
        colors=3 * ['blue'],
        linestyles=['solid', 'dashed', 'dotted']
    )

    # Customize left figure axes.
    customize_axis(
        ax=ax[T_PLOT],
        x_title='',
        y_title=r'$\displaystyle T(\tau, x)$ [GeV]',
        no_xnums=True,
        ylim=(0.02, 0.23)
    )
    customize_axis(
        ax=ax[MU_PLOT],
        x_title='',
        y_title=r'$\displaystyle\mu_Y(\tau, x)$ [GeV]',
        no_xnums=True
    )
    customize_axis(
        ax=ax[PIYY_PLOT],
        x_title=r'$\displaystyle x$ [fm]',
        y_title=r'$\displaystyle\pi^{yy}(\tau, x) / w(\tau, x)$'
    )
    customize_axis(
        ax=ax[PIXY_PLOT],
        x_title=r'$\displaystyle x$ [fm]',
        y_title=r'$\displaystyle\pi^{xy}(\tau, x) / w(\tau, x)$',
        ylim=(-0.85, 0.1)
    )
    ax[T_PLOT].legend(loc='upper right', fontsize=10)
    ax[MU_PLOT].legend(loc='upper right', fontsize=10)
    ax[T_PLOT].text(
        0.12, 0.83, r'\textsc{EoS1}',
        transform=ax[T_PLOT].transAxes,
        fontsize=10,
        bbox={'facecolor': 'white', 'boxstyle': 'round', 'linewidth': 0.5},
        horizontalalignment='center'
    )
    for plot_idx, label in zip([T_PLOT, MU_PLOT, PIYY_PLOT, PIXY_PLOT], ['a', 'b', 'c', 'd']):
        ax[plot_idx].text(
            0.07, 0.93, f'({label})',
            transform=ax[plot_idx].transAxes,
            fontsize=10,
            horizontalalignment='center'
        )
    fig.tight_layout()
    logger.info(f"Saving left figure to {args.output1}")
    fig.savefig(args.output1)

    # Customize right figure axes.
    customize_axis(
        ax=ax2[E_PLOT],
        x_title=r'$\displaystyle x$ [fm]',
        y_title=r'$\displaystyle\mathcal E(\tau, x)$ [GeV/fm$^{3}$]'
    )
    ax2[E_PLOT].set_yscale('log')
    customize_axis(
        ax=ax2[N_PLOT],
        x_title=r'$\displaystyle x$ [fm]',
        y_title=r'$\displaystyle n_Y(\tau, x)$ [fm$^{-3}$]',
        ylim=(0.5, 2000)
    )
    ax2[N_PLOT].set_yscale('log')
    customize_axis(
        ax=ax2[S_PLOT],
        x_title=r'$\displaystyle x$ [fm]',
        y_title=r'$\displaystyle s(\tau, x)$ [fm$^{-3}$]'
    )
    ax2[S_PLOT].set_yscale('log')
    ax2[E_PLOT].legend(loc='lower center', fontsize=10)
    ax2[N_PLOT].legend(loc='lower center', fontsize=10)
    ax2[E_PLOT].text(
        0.12, 0.83, r'\textsc{EoS1}',
        transform=ax2[E_PLOT].transAxes,
        fontsize=10,
        bbox={'facecolor': 'white', 'boxstyle': 'round', 'linewidth': 0.5},
        horizontalalignment='center'
    )
    for plot_idx, label in zip([E_PLOT, N_PLOT, S_PLOT], ['a', 'b', 'c']):
        ax2[plot_idx].text(
            0.07, 0.93, f'({label})',
            transform=ax2[plot_idx].transAxes,
            fontsize=10,
            horizontalalignment='center'
        )
    fig2.tight_layout()
    logger.info(f"Saving right figure to {args.output2}")
    fig2.savefig(args.output2)

    logger.info("Plotting complete.")


if __name__ == "__main__":
    main()
