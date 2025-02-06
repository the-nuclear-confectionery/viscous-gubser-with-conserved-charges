#!/usr/bin/env python3
"""
This module solves and plots the evolution of various physical quantities for a 
massless quark-gluon plasma using a given equation of state. The evolution equations 
are integrated using SciPy's ODE solver and then interpolated to obtain the Milne 
variables. Results are plotted on two figures: one for temperature, chemical potential, 
and shear components, and another for energy density, number density, and entropy density.
"""

import sys
# Append parent directory to sys.path to import local modules
sys.path.append('..')

# Standard scientific computing imports
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import List

# Custom modules (ensure these are in the Python path)
from plots.plotting_settings import *
from system_massless_qgp import MasslessQGP
from variable_conversions import milne_T, milne_mu

# Define constant indices for subplot axes.
# For the left 2x2 subplot grid (ax), indices are given as (row, col).
T_PLOT = (0, 0)      # Temperature plot index (row 0, col 0)
MU_PLOT = (0, 1)     # Chemical potential plot index (row 0, col 1)
PIYY_PLOT = (1, 0)   # Shear stress component (pi^yy normalized) plot index (row 1, col 0)
PIXY_PLOT = (1, 1)   # Shear stress component (pi^xy normalized) plot index (row 1, col 1)

# For the right subplot grid (ax2), which is 1D:
E_PLOT = 0           # Energy density plot index
N_PLOT = 1           # Number density plot index
S_PLOT = 2           # Entropy density plot index

# Initialize the system (an instance of MasslessQGP)
system = MasslessQGP()


def solve_and_plot(
    ax_1: np.ndarray,
    ax_2: np.ndarray,
    y0s: np.ndarray,
    rhos_1: np.ndarray,
    rhos_2: np.ndarray,
    xs: np.ndarray,
    taus: np.ndarray,
    color: List[str],
    linestyle: List[str],
    add_labels: bool = False,
) -> None:
    """
    Solve the evolution equations for the system and plot the results on the given axes.

    Parameters:
        ax_1 (np.ndarray): 2D array of matplotlib axes for temperature, chemical potential,
                           and shear components.
        ax_2 (np.ndarray): 1D array of matplotlib axes for energy density, number density, and entropy.
        y0s (np.ndarray): Initial condition array [T0, mu0, pi0].
        rhos_1 (np.ndarray): Array of negative rho values (reversed later).
        rhos_2 (np.ndarray): Array of positive rho values.
        xs (np.ndarray): Spatial grid for plotting.
        taus (np.ndarray): Array of proper time values at which to evaluate the Milne variables.
        color (List[str]): List of color strings for plotting different tau curves.
        linestyle (List[str]): List of line style strings for plotting different tau curves.
        add_labels (bool): Flag indicating whether to add labels to the plots.
    """
    # Solve the equations of motion for two different ranges of 'rho'
    soln_1 = odeint(system.eom.for_scipy, y0s, rhos_1)
    soln_2 = odeint(system.eom.for_scipy, y0s, rhos_2)

    # Concatenate solutions: reverse the first part and then append the second part.
    # Each column corresponds to a physical variable.
    t_hat = np.concatenate((soln_1[:, 0][::-1], soln_2[:, 0]))
    mu_hat = np.concatenate((soln_1[:, 1][::-1], soln_2[:, 1]))
    pi_bar_hat = np.concatenate((soln_1[:, 2][::-1], soln_2[:, 2]))
    rhos = np.concatenate((rhos_1[::-1], rhos_2))

    # Create interpolation functions for the temperature, chemical potential, and shear variable.
    t_interp = interp1d(rhos, t_hat)
    mu_interp = interp1d(rhos, mu_hat)
    pi_interp = interp1d(rhos, pi_bar_hat)

    # Loop over each proper time tau to compute and plot the Milne evolution.
    for n, tau in enumerate(taus):
        # Compute the Milne-evolved temperature and chemical potential
        t_evol = milne_T(tau, xs, 1, t_interp)
        mu_evol = milne_mu(tau, xs, 1, mu_interp)

        # Compute the energy density, number density, and entropy density in the Milne frame.
        # The constants (0.0 and 1.0) are parameters specific to the model.
        e_evol = system.milne_energy(tau, xs, 0.0, 1.0, t_interp, mu_interp)
        n_evol = system.milne_number(tau, xs, 0.0, 1.0, t_interp, mu_interp)
        s_evol = system.milne_entropy(tau, xs, 0.0, 1.0, t_interp, mu_interp)

        # Plot temperature evolution on ax_1[T_PLOT]
        # Only add a label for the first tau value to indicate the initial condition ratio.
        ax_1[T_PLOT].plot(
            xs,
            t_evol,
            color=color[n],
            lw=1,
            ls=linestyle[n],
            label=(
                r'$\displaystyle\hat{\mu}_{Y,\,0}/\hat{T}_0=' + f'{y0s[1] / y0s[0]:.1f}$'
                if n == 0 else None
            )
        )

        # Plot chemical potential evolution on ax_1[MU_PLOT]
        ax_1[MU_PLOT].plot(
            xs,
            mu_evol,
            color=color[n],
            lw=1,
            ls=linestyle[n],
            label=(r'$\displaystyle\tau = ' + f'{tau:.2f}' + r'$ [fm/$c$]' if add_labels else None)
        )

        # Compute shear stress tensor components in the Milne frame.
        # 'nonzero_xy=True' ensures that off-diagonal components are computed.
        pi_xx, pi_yy, pi_xy, pi_nn = system.milne_pi(
            tau,
            xs,
            0.0,
            1,
            t_interp,
            mu_interp,
            pi_interp,
            nonzero_xy=True,
        )

        # Normalize the shear stress components by the enthalpy (4e/3) and plot.
        ax_1[PIYY_PLOT].plot(
            xs,
            pi_yy / (4.0 * e_evol / 3.0),
            color=color[n],
            lw=1,
            ls=linestyle[n]
        )
        ax_1[PIXY_PLOT].plot(
            xs,
            pi_xy / (4.0 * e_evol / 3.0),
            color=color[n],
            lw=1,
            ls=linestyle[n]
        )

        # Plot energy density, number density, and entropy density on ax_2.
        # For energy density and number density, add labels if specified.
        ax_2[E_PLOT].plot(
            xs,
            e_evol,
            color=color[n],
            lw=1,
            ls=linestyle[n],
            label=(
                r'$\displaystyle\hat{\mu}_{Y,\,0}/\hat{T}_0=' + f'{y0s[1] / y0s[0]:.1f}$'
                if n == 0 else None
            )
        )
        ax_2[N_PLOT].plot(
            xs,
            n_evol,
            color=color[n],
            lw=1,
            ls=linestyle[n],
            label=(r'$\displaystyle\tau=' + f'{tau:.2f}$ [fm/$c$]' if add_labels else None)
        )
        ax_2[S_PLOT].plot(
            xs,
            s_evol,
            color=color[n],
            lw=1,
            ls=linestyle[n]
        )


def main():
    """
    Set up figures and axes, compute the evolution for different initial conditions,
    and save the generated plots to PDF files.
    """
    # Create the first figure with a 2x2 grid of subplots for T, mu, and shear components.
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 6), dpi=1200, sharex=True)
    fig.patch.set_facecolor('white')

    # Create the second figure with 1x3 subplots for energy, number, and entropy densities.
    fig2, ax2 = plt.subplots(nrows=1, ncols=3, figsize=(10.5, 3.36), dpi=1200)

    # Define the spatial grid and rho ranges.
    xs = np.linspace(-6, 6, 200)
    rhos_1 = np.linspace(-10, 0, 1000)[::-1]  # Reverse order for negative rhos
    rhos_2 = np.linspace(0, 10, 1000)

    # ----------------------------
    # First set of initial conditions (EoS1) with low chemical potential
    # ----------------------------
    y0s = np.array([1.2, 1e-20 * 1.2, 0.0])
    solve_and_plot(
        ax_1=ax,
        ax_2=ax2,
        y0s=y0s,
        rhos_1=rhos_1,
        rhos_2=rhos_2,
        xs=xs,
        taus=np.array([1.2, 2.0, 3.0]),
        color=3 * ['black'],
        linestyle=['solid', 'dashed', 'dotted'],
        add_labels=True,
    )

    # ----------------------------
    # Second set of initial conditions with a moderate chemical potential
    # ----------------------------
    y0s = np.array([1.2, 5 * 1.2, 0.0])
    solve_and_plot(
        ax_1=ax,
        ax_2=ax2,
        y0s=y0s,
        rhos_1=rhos_1,
        rhos_2=rhos_2,
        xs=xs,
        taus=np.array([1.2, 2.0, 3.0]),
        color=3 * ['red'],
        linestyle=['solid', 'dashed', 'dotted'],
    )

    # ----------------------------
    # Third set of initial conditions with a higher chemical potential
    # ----------------------------
    y0s = np.array([1.2, 8.0 * 1.2, 0.0])
    solve_and_plot(
        ax_1=ax,
        ax_2=ax2,
        y0s=y0s,
        rhos_1=rhos_1,
        rhos_2=rhos_2,
        xs=xs,
        taus=np.array([1.2, 2.0, 3.0]),
        color=3 * ['blue'],
        linestyle=['solid', 'dashed', 'dotted']
    )

    # ----------------------------
    # Customize axes for the left figure (ax: 2x2 grid)
    # ----------------------------
    customize_axis(
        ax=ax[T_PLOT],
        x_title=r'',
        y_title=r'$\displaystyle T(\tau, x)$ [GeV]',
        no_xnums=True,
        ylim=(0.02, 0.23)
    )
    
    customize_axis(
        ax=ax[MU_PLOT],
        x_title=r'',
        y_title=r'$\displaystyle\mu_Y(\tau, x)$ [GeV]',
        no_xnums=True,
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

    # Add legends to the first two subplots
    ax[T_PLOT].legend(loc='upper right', fontsize=10)
    ax[MU_PLOT].legend(loc='upper right', fontsize=10)

    # Add a text label indicating the equation of state ("EoS1") to the Temperature plot.
    ax[T_PLOT].text(
        0.12, 0.83, "EoS1",
        transform=ax[T_PLOT].transAxes,
        fontsize=10,
        bbox={'facecolor': 'white', 'boxstyle': 'round', 'linewidth': 0.5},
        horizontalalignment='center'
    )

    # Add subplot labels (a, b, c, d) to each panel of the left figure.
    for plot_idx, label in zip([T_PLOT, MU_PLOT, PIYY_PLOT, PIXY_PLOT], ['a', 'b', 'c', 'd']):
        ax[plot_idx].text(
            0.07, 0.93,
            f'({label})',
            transform=ax[plot_idx].transAxes,
            fontsize=10,
            horizontalalignment='center'
        )

    # Adjust layout and save the left figure.
    fig.tight_layout()

    fig_name = './output/Fig1_VGCC-T_mu_pi.pdf'
    print(f'Saving figure to file {fig_name}')
    fig.savefig(fig_name)

    # ----------------------------
    # Customize axes for the right figure (ax2: 1x3 grid)
    # ----------------------------
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
        ylim=(5*1e-1, 2*1e3)
    )
    ax2[N_PLOT].set_yscale('log')

    customize_axis(
        ax=ax2[S_PLOT],
        x_title=r'$\displaystyle x$ [fm]',
        y_title=r'$\displaystyle s(\tau, x)$ [fm$^{-3}$]'
    )
    ax2[S_PLOT].set_yscale('log')

    # Add legends to the energy and number density plots.
    ax2[E_PLOT].legend(loc='lower center', fontsize=10)
    ax2[N_PLOT].legend(loc='lower center', fontsize=10)

    # Add a text label ("EoS1") to the energy density plot.
    ax2[E_PLOT].text(
        0.12, 0.83, "EoS1",
        transform=ax2[E_PLOT].transAxes,
        fontsize=10,
        bbox={'facecolor': 'white', 'boxstyle': 'round', 'linewidth': 0.5},
        horizontalalignment='center'
    )

    # Add subplot labels to the right figure.
    # Note: Only three labels are needed for the three subplots.
    for plot_idx, label in zip([E_PLOT, N_PLOT, S_PLOT], ['a', 'b', 'c']):
        ax2[plot_idx].text(
            0.07, 0.93,
            f'({label})',
            transform=ax2[plot_idx].transAxes,
            fontsize=10,
            horizontalalignment='center'
        )

    # Adjust layout and save the right figure.
    fig2.tight_layout()
    fig_name = './output/Fig2_VGCC-e_n_s.pdf'
    print(f'Saving figure to file {fig_name}')
    fig2.savefig(fig_name)


if __name__ == "__main__":
    main()
