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

from typing import Dict, Any, Union
from pathlib import Path

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import List

# Append parent directory so we can import our local modules.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from eom.eom_factory import get_eom
from plots.plotting_settings import (
    customize_axis,
)  # Assumed to be defined in your project
from utils.analytic_functions import (
    milne_T,
    milne_mu,
    milne_energy,
    milne_number,
    milne_entropy,
    milne_pi,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Config:
    """
    Configuration object that holds simulation parameters loaded from a YAML file.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.tau_0: float = config.get("tau_0", 1.0)
        self.tau_f: float = config.get("tau_f", 1.6)
        self.tau_step: float = config.get("tau_step", 0.10)
        self.x_min: float = config.get("x_min", -5.0)
        self.x_max: float = config.get("x_max", 5.0)
        self.x_step: float = config.get("x_step", 0.05)
        self.y_min: float = config.get("y_min", -5.0)
        self.y_max: float = config.get("y_max", 5.0)
        self.y_step: float = config.get("y_step", 0.05)
        self.eta_min: float = config.get("eta_min", -0.1)
        self.eta_max: float = config.get("eta_max", 0.1)
        self.step_eta: float = config.get("stepEta", 0.1)
        self.T_hat_0: float = config.get("T_hat_0", 1.20)
        self.pi_bar_hat_0: float = config.get("pi_bar_hat_0", 0.0)
        self.muB_hat_0: float = config.get("muB_hat_0", 1e-20)
        self.muS_hat_0: float = config.get("muS_hat_0", 1e-20)
        self.muQ_hat_0: float = config.get("muQ_hat_0", 1e-20)
        self.T_ast: float = config.get("T_ast", 1.000)
        self.muB_ast: float = config.get("muB_ast", 1.00)
        self.muS_ast: float = config.get("muS_ast", 1.00)
        self.muQ_ast: float = config.get("muQ_ast", 1.00)
        self.ETA_OVER_S: float = config.get("eta_over_s", 0.20)
        self.CTAUR: float = config.get("CTAUR", 5.0)
        self.q: float = config.get("q", 1.0)
        self.output_dir: str = config.get(
            "output_dir", "data/output/analytic_solutions"
        )

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        """
        Create a Config instance from a YAML configuration file.

        Parameters:
            path (str): Path to the YAML configuration file.

        Returns:
            Config: The configuration object.
        """
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return cls(config)


# Define subplot indices (for the left figure arranged in a 2x2 grid)
T_PLOT = (0, 0)  # Temperature plot
MU_PLOT = (0, 1)  # Chemical potential plot
PIYY_PLOT = (1, 0)  # Shear component (pi^yy normalized) plot
PIXY_PLOT = (1, 1)  # Shear component (pi^xy normalized) plot

# For the right figure with 1x3 grid:
E_PLOT = 0  # Energy density
N_PLOT = 1  # Number density
S_PLOT = 2  # Entropy density


def setup_initial_conditions(cfg: Config) -> np.ndarray:
    """
    Collect de Sitter initial conditions from the configuration file.

    Parameters:
        cfg (Config): The configuration object.

    Returns:
        np.ndarray: An array of initial conditions.
    """
    logger.debug("Setting up initial conditions from configuration file.")
    return np.array(
        [cfg.T_hat_0, cfg.muB_hat_0, cfg.muS_hat_0, cfg.muQ_hat_0, cfg.pi_bar_hat_0]
    )


def setup_eos_parameters(cfg: Config) -> Dict[str, Union[float, np.ndarray]]:
    """
    Setup the Equation of State (EoS) parameters based on the configuration.

    Parameters:
        cfg (Config): The configuration object.

    Returns:
        Dict[str, Union[float, np.ndarray]]: Dictionary of EoS parameters.
    """
    logger.debug("Setting up EoS parameters from configuration file.")
    eos_params = {
        "T_ast": cfg.T_ast,
        "mu_ast": np.array([cfg.muB_ast, cfg.muS_ast, cfg.muQ_ast]),
    }
    return eos_params


def setup_eom_parameters(cfg: Config) -> Dict[str, float]:
    """
    Setup the equations of motion (EoM) parameters based on the configuration.

    Parameters:
        cfg (Config): The configuration object.

    Returns:
        Dict[str, float]: Dictionary of EoM parameters.
    """
    logger.debug("Setting up EoM parameters from configuration file.")
    eom_params = {"CTAUR": cfg.CTAUR, "ETA_OVER_S": cfg.ETA_OVER_S}
    return eom_params


def solve_equations(initial_conditions: np.ndarray, eom_instance: Any) -> tuple:
    """
    Solve the equations of motion using the provided EoM instance.
    Returns interpolation functions for T, mu, and pi.

    Parameters:
        initial_conditions (np.ndarray): Initial condition values.
        eom_instance (Any): Instance providing the EoM and related functions.

    Returns:
        tuple: (T_interp, mu_interp, pi_interp) where:
            - T_hat_interp is an interpolation function for T_hat,
            - mu_hat_interp is a list of interpolation functions for each mu component,
            - pi_bar_hat_interp is an interpolation function for pi_bar_hat.
    """
    logger.info("Solving equations of motion.")
    # Separate the negative and positive values of rho.
    rhos_neg = np.linspace(-10, 0, 1000)[::-1]
    rhos_pos = np.linspace(0, 10, 1000)
    # Solve the equations of motion for the negative and positive values of rho.
    soln_neg = odeint(eom_instance.eom, initial_conditions, rhos_neg)
    soln_pos = odeint(eom_instance.eom, initial_conditions, rhos_pos)
    # Merge the solutions for negative and positive values of rho.
    T_hat = np.concatenate((soln_neg[:, 0][::-1], soln_pos[:, 0]))
    mu_hat = [np.concatenate((soln_neg[:, i][::-1], soln_pos[:, i])) for i in [1, 2, 3]]
    pi_bar_hat = np.concatenate((soln_neg[:, 4][::-1], soln_pos[:, 4]))
    rhos = np.concatenate((rhos_neg[::-1], rhos_pos))
    # Interpolate the solutions.
    T_hat_interp = interp1d(rhos, T_hat)
    mu_hat_interp = [interp1d(rhos, f) for f in mu_hat]
    pi_bar_hat_interp = interp1d(rhos, pi_bar_hat)
    # Return the interpolation functions.
    logger.info("Equations of motion solved.")
    return T_hat_interp, mu_hat_interp, pi_bar_hat_interp


def plot(
    cfg: Config,
    ax1: np.ndarray,
    ax2: np.ndarray,
    solutions: tuple,
    eom_instance: Any,
    initial_conditions: np.ndarray,
    xs: np.ndarray,
    taus: np.ndarray,
    colors: List[str],
    linestyles: List[str],
    add_labels: bool = False,
) -> None:
    """
    Plot the evolution of temperature, chemical potential, shear components, energy density,
    number density, and entropy density.
    """
    logger.info("Plotting the evolution of the system.")

    # Unpack the solutions.
    T_hat_interp, mu_hat_interp, pi_bar_hat_interp = solutions

    # Loop over the time steps.
    for n, tau in enumerate(taus):
        temperature = milne_T(tau=tau, r=xs, q=cfg.q, interpolated_T_hat=T_hat_interp)
        chemical_potential = milne_mu(
            tau=tau, r=xs, q=cfg.q, interpolated_mu_hat=mu_hat_interp[0]
        )
        energy = milne_energy(
            tau=tau,
            x=xs,
            y=0.0,
            q=cfg.q,
            interpolated_T_hat=T_hat_interp,
            interpolated_mu_hat=mu_hat_interp,
            eos_instance=eom_instance.eos,
        )
        number_density = milne_number(
            tau=tau,
            x=xs,
            y=0.0,
            q=cfg.q,
            interpolated_T_hat=T_hat_interp,
            interpolated_mu_hat=mu_hat_interp,
            eos_instance=eom_instance.eos,
        )
        # # If number_density has shape (3, 200), select the first component.
        # if number_density.ndim > 1:
        #     number_density = number_density[0, :]
        entropy_density = milne_entropy(
            tau=tau,
            x=xs,
            y=0.0,
            q=cfg.q,
            interpolated_T_hat=T_hat_interp,
            interpolated_mu_hat=mu_hat_interp,
            eos_instance=eom_instance.eos,
        )
        number_density = (
            (4 / 3) * energy - temperature * entropy_density
        ) / chemical_potential
        pi_xx, pi_yy, pi_xy, _ = milne_pi(
            tau=tau,
            x=xs,
            y=0.0,
            q=cfg.q,
            interpolated_T_hat=T_hat_interp,
            interpolated_mu_hat=mu_hat_interp,
            interpolated_pi_bar_hat=pi_bar_hat_interp,
            eos_instance=eom_instance.eos,
            nonzero_xy=True,
        )

        # Plot temperature
        logger.debug(f"Plotting temperature for tau={tau:.2f} fm/c")
        ax1[T_PLOT].plot(
            xs,
            temperature,
            color=colors[n],
            lw=1,
            ls=linestyles[n],
            label=(
                r"$\displaystyle\hat{\mu}_{Y,\,0}/\hat{T}_0="
                + f"{initial_conditions[1] / initial_conditions[0]:.1f}$"
                if n == 0
                else None
            ),
        )
        # Plot chemical potential
        logger.debug(f"Plotting chemical potential for tau={tau:.2f} fm/c")
        ax1[MU_PLOT].plot(
            xs,
            chemical_potential,
            color=colors[n],
            lw=1,
            ls=linestyles[n],
            label=(
                r"$\displaystyle \tau = " + f"{tau:.2f}$ fm/$c$" if add_labels else None
            ),
        )
        # Plot shear components
        logger.debug(f"Plotting shear components for tau={tau:.2f} fm/c")
        ax1[PIYY_PLOT].plot(
            xs, pi_yy / (4.0 * energy / 3.0), color=colors[n], lw=1, ls=linestyles[n]
        )
        ax1[PIXY_PLOT].plot(
            xs, pi_xy / (4.0 * energy / 3.0), color=colors[n], lw=1, ls=linestyles[n]
        )

        # Plot energy density
        logger.debug(f"Plotting energy density for tau={tau:.2f} fm/c")
        ax2[E_PLOT].plot(
            xs,
            energy,
            color=colors[n],
            lw=1,
            ls=linestyles[n],
            label=(
                r"$\displaystyle\hat{\mu}_{Y,\,0}/\hat{T}_0="
                + f"{initial_conditions[1] / initial_conditions[0]:.1f}$"
                if n == 0
                else None
            ),
        )
        # Plot number density
        logger.debug(f"Plotting number density for tau={tau:.2f} fm/c")
        ax2[N_PLOT].plot(
            xs,
            number_density,
            color=colors[n],
            lw=1,
            ls=linestyles[n],
            label=(
                r"$\displaystyle \tau = " + f"{tau:.2f}$ fm/$c$" if add_labels else None
            ),
        )
        # Plot entropy density
        logger.debug(f"Plotting entropy density for tau={tau:.2f} fm/c")
        ax2[S_PLOT].plot(xs, entropy_density, color=colors[n], lw=1, ls=linestyles[n])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Plot Fig1: VGCC evolution for T, μ, shear components,
                       energy, number & entropy densities.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Developed by the UIUC Nuclear Theory Group.",
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the config YAML file"
    )
    parser.add_argument(
        "--eos",
        default="EoS1",
        choices=["EoS1", "EoS2"],
        help="type of equation of state to use",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output file for the left figure (T, μ, shear)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the script.
    """
    # Parse command-line arguments.
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Load the configuration file.
    logger.debug(f"Loading configuration from {args.config}.")
    try:
        cfg = Config.from_yaml(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration - {e}.")
        sys.exit(1)

    # Create output directory
    dir_path = Path(args.output_path).absolute()
    dir_path.mkdir(parents=True, exist_ok=True)

    # Setup EoS and EoM parameters.
    eos_params = setup_eos_parameters(cfg)
    eom_params = setup_eom_parameters(cfg)

    # Create the corresponding EoM instance using the selected EoS type.
    try:
        eom_instance = get_eom(args.eos, eos_params=eos_params, eom_params=eom_params)
    except ValueError as exc:
        logger.error(f"EoM are not available - {exc}.")
        sys.exit(1)

    # Define spatial grid and rho ranges.
    xs = np.linspace(-6, 6, 200)

    # Prepare figure 1 (2x2 grid) and figure 2 (1x3 grid)
    fig, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(7, 6), dpi=1200, sharex=True)
    fig.patch.set_facecolor("white")
    fig2, ax2 = plt.subplots(nrows=1, ncols=3, figsize=(10.5, 3.36), dpi=1200)

    # First set of initial conditions (low μ)
    # Setup initial conditions.
    initial_conditions = np.array([cfg.T_hat_0, 0.0, 0.0, 0.0, cfg.pi_bar_hat_0])
    # Solve the equations of motion.
    solutions = solve_equations(initial_conditions, eom_instance)
    plot(
        cfg,
        ax1,
        ax2,
        solutions,
        eom_instance,
        initial_conditions,
        xs,
        taus=np.array([1.0, 1.2, 2.0]),
        colors=3 * ["black"],
        linestyles=["solid", "dashed", "dotted"],
        add_labels=True,
    )

    # Second set (moderate μ)
    # Setup initial conditions.
    initial_conditions = np.array(
        [cfg.T_hat_0, 10 * cfg.T_hat_0, 0.0, 0.0, cfg.pi_bar_hat_0]
    )
    # Solve the equations of motion.
    solutions = solve_equations(initial_conditions, eom_instance)
    plot(
        cfg,
        ax1,
        ax2,
        solutions,
        eom_instance,
        initial_conditions,
        xs,
        taus=np.array([1.0, 1.2, 2.0]),
        colors=3 * ["red"],
        linestyles=["solid", "dashed", "dotted"],
        add_labels=False,
    )

    # Third set (high μ)
    # Setup initial conditions.
    initial_conditions = np.array(
        [cfg.T_hat_0, 15 * cfg.T_hat_0, 0.0, 0.0, cfg.pi_bar_hat_0]
    )
    # Solve the equations of motion.
    solutions = solve_equations(initial_conditions, eom_instance)
    plot(
        cfg,
        ax1,
        ax2,
        solutions,
        eom_instance,
        initial_conditions,
        xs,
        taus=np.array([1.0, 1.2, 2.0]),
        colors=3 * ["blue"],
        linestyles=["solid", "dashed", "dotted"],
        add_labels=False,
    )

    # Customize left figure axes.
    customize_axis(
        ax=ax1[T_PLOT],
        x_title="",
        y_title=r"$\displaystyle T(\tau, x)$ [GeV]",
        no_xnums=False,
        ylim=(-0.02, 0.42),
    )
    customize_axis(
        ax=ax1[MU_PLOT],
        x_title="",
        y_title=r"$\displaystyle\mu_Y(\tau, x)$ [GeV]",
        no_xnums=False,
    )
    customize_axis(
        ax=ax1[PIYY_PLOT],
        x_title=r"$\displaystyle x$ [fm]",
        y_title=r"$\displaystyle\pi^{yy}(\tau, x) / w(\tau, x)$",
        xlim=(-6, 6),
    )
    customize_axis(
        ax=ax1[PIXY_PLOT],
        x_title=r"$\displaystyle x$ [fm]",
        y_title=r"$\displaystyle\pi^{xy}(\tau, x) / w(\tau, x)$",
        ylim=(-0.35, 0.05),
        xlim=(-6, 6),
    )
    ax1[T_PLOT].legend(loc="upper right", fontsize=10)
    ax1[MU_PLOT].legend(loc="upper right", fontsize=10)
    ax1[T_PLOT].text(
        0.12,
        0.83,
        r"\textsc{EoS1}",
        transform=ax1[T_PLOT].transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "boxstyle": "round", "linewidth": 0.5},
        horizontalalignment="center",
    )
    for plot_idx, label in zip(
        [T_PLOT, MU_PLOT, PIYY_PLOT, PIXY_PLOT], ["a", "b", "c", "d"]
    ):
        ax1[plot_idx].text(
            0.07,
            0.93,
            f"({label})",
            transform=ax1[plot_idx].transAxes,
            fontsize=10,
            horizontalalignment="center",
        )

    file_name = "Fig1_VGCC-T_mu_pi.pdf"
    file_path = dir_path / file_name
    fig.tight_layout()
    logger.info(f"Saving first figure to {file_path}")
    fig.savefig(file_path)

    # Customize right figure axes.
    customize_axis(
        ax=ax2[E_PLOT],
        x_title=r"$\displaystyle x$ [fm]",
        y_title=r"$\displaystyle\mathcal E(\tau, x)$ [GeV/fm$^{3}$]",
    )
    ax2[E_PLOT].set_yscale("log")
    customize_axis(
        ax=ax2[N_PLOT],
        x_title=r"$\displaystyle x$ [fm]",
        y_title=r"$\displaystyle n_Y(\tau, x)$ [fm$^{-3}$]",
        ylim=(5 * 1e-2, 2 * 1e2),
    )
    ax2[N_PLOT].set_yscale("log")
    customize_axis(
        ax=ax2[S_PLOT],
        x_title=r"$\displaystyle x$ [fm]",
        y_title=r"$\displaystyle s(\tau, x)$ [fm$^{-3}$]",
    )
    ax2[S_PLOT].set_yscale("log")
    ax2[E_PLOT].legend(loc="lower center", fontsize=10)
    ax2[N_PLOT].legend(loc="lower center", fontsize=10)
    ax2[E_PLOT].text(
        0.12,
        0.83,
        r"\textsc{EoS1}",
        transform=ax2[E_PLOT].transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "boxstyle": "round", "linewidth": 0.5},
        horizontalalignment="center",
    )
    for plot_idx, label in zip([E_PLOT, N_PLOT, S_PLOT], ["a", "b", "c"]):
        ax2[plot_idx].text(
            0.07,
            0.93,
            f"({label})",
            transform=ax2[plot_idx].transAxes,
            fontsize=10,
            horizontalalignment="center",
        )
    fig2.tight_layout()
    file_name = "Fig2_VGCC-e_n_s.pdf"
    file_path = dir_path / file_name

    logger.info(f"Saving second figure to {file_path}")
    fig2.savefig(file_path)

    logger.info("Plotting complete.")


if __name__ == "__main__":
    main()
