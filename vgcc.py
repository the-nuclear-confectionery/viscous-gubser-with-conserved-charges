#!/usr/bin/env python3
"""
VISCOUS GUBSER FLOW WITH CONSERVED CHARGES (VGCC)

This script solves the equations of motion for viscous Gubser flow with
conserved charges. It reads configuration from a YAML file, sets up initial
conditions and parameters, solves the equations, and outputs the results.
"""

import sys
import logging
import argparse

from typing import Union, Dict, Any

import yaml
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.integrate import odeint

from utils.constants import print_header, HBARC, tolerance
from utils.analytic_functions import u_x, u_y, milne_energy, milne_number, milne_pi
from eom.eom_factory import get_eom

# Configure logging
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


def setup_initial_conditions(cfg: Config) -> np.ndarray:
    """
    Collect de Sitter initial conditions from the configuration file.

    Parameters:
        cfg (Config): The configuration object.

    Returns:
        np.ndarray: An array of initial conditions.
    """
    logger.debug(f"Setting up initial conditions from configuration file.")
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
    logger.debug(f"Setting up EoS parameters from configuration file.")
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
    logger.debug(f"Setting up EoM parameters from configuration file.")
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
    logger.info(f"Solving equations of motion.")
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
    logger.info(f"Equations of motion solved.")
    return T_hat_interp, mu_hat_interp, pi_bar_hat_interp


def output_solution(
    cfg: Config, solutions: tuple, eom_instance: Any, mode: str
) -> None:
    """
    Generate output files for either initial conditions (single time step) or the
    entire evolution (multiple time steps).

    In 'initial_condition' mode, only the first time step (tau = cfg.tau_0) is
    generated and saved to 'VGCC_initial_condition.dat'. Otherwise, each time step
    is written to its own file using the naming convention 'VGCC_tau={tau:.2f}.dat'.

    Parameters:
        cfg (Config): The configuration object.
        solutions (tuple): Tuple containing interpolation function for t_hat, mu_hat, and pi_bar_hat.
        eom_instance (Any): Instance providing EoM-related functions.
        mode (str): Operation mode; either "initial_condition" or "evolution".
    """
    logger.info(f"Generating ouput for {mode} mode.")

    # Unpack the solutions.
    T_hat_interp, mu_hat_interp, pi_bar_hat_interp = solutions

    # Create output directory
    dir_path = Path(cfg.output_dir).absolute()
    dir_path.mkdir(parents=True, exist_ok=True)

    # Determine the time steps based on the mode.
    time_steps = (
        [cfg.tau_0]
        if mode == "initial_condition"
        else np.arange(cfg.tau_0, cfg.tau_f, cfg.tau_step)
    )

    # Use tqdm to display progress bar.
    iterator = tqdm(
        time_steps,
        desc="Progress",
        position=0,
        leave=False,
        bar_format="{l_bar}{bar:40} | [Elapsed:{elapsed} | Remaining:{remaining}]",
        colour="green",
    )

    # Loop over each time step.
    for tau in iterator:
        rows = []
        # Loop over x and y coordinates
        for x in np.arange(cfg.x_min, cfg.x_max + cfg.x_step, cfg.x_step):
            for y in np.arange(cfg.y_min, cfg.y_max + cfg.y_step, cfg.y_step):
                energy = milne_energy(
                    tau=tau,
                    x=x,
                    y=y,
                    q=cfg.q,
                    interpolated_T_hat=T_hat_interp,
                    interpolated_mu_hat=mu_hat_interp,
                    eos_instance=eom_instance.eos,
                )
                nums = milne_number(
                    tau=tau,
                    x=x,
                    y=y,
                    q=cfg.q,
                    interpolated_T_hat=T_hat_interp,
                    interpolated_mu_hat=mu_hat_interp,
                    eos_instance=eom_instance.eos,
                )
                pis = milne_pi(
                    tau=tau,
                    x=x,
                    y=y,
                    q=cfg.q,
                    interpolated_T_hat=T_hat_interp,
                    interpolated_mu_hat=mu_hat_interp,
                    interpolated_pi_bar_hat=pi_bar_hat_interp,
                    eos_instance=eom_instance.eos,
                )
                row = [
                    x,
                    y,
                    0,  # eta
                    energy,
                    nums[0],
                    nums[1],
                    nums[2],
                    u_x(tau, x, y, cfg.q),
                    u_y(tau, x, y, cfg.q),
                    0,  # u_eta
                    0,  # bulk
                    pis[0],
                    pis[2],
                    0,
                    pis[1],
                    0,
                    pis[3],
                ]
                rows.append(row)

        # Temporarily store output in a DataFrame.
        columns = [
            "x",
            "y",
            "eta",
            "energy",
            "num1",
            "num2",
            "num3",
            "u_x",
            "u_y",
            "u_eta",
            "bulk",
            "pi_xx",
            "pi_xy",
            "pi_xeta",
            "pi_yy",
            "pi_yeta",
            "pi_etaeta",
        ]
        df = pd.DataFrame(rows, columns=columns)
        header_str = f"#0 {cfg.x_step} {cfg.y_step} {cfg.step_eta} 0 {cfg.x_min} {cfg.y_min} {cfg.eta_min}\n"
        # Define formatters for the output.
        fmt = {col: (lambda x: f"{x:12.6f}") for col in columns}
        out_str = df.to_string(index=False, header=False, formatters=fmt)

        # Choose the file name based on mode and time step.
        if mode == "initial_condition":
            file_name = "VGCC_initial_condition.dat"
        else:
            file_name = f"VGCC_tau={tau:.2f}.dat"

        # Write the output to a file.
        file_path = dir_path / file_name
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(header_str)
            f.write(out_str + "\n")
    logger.info(f"Output written to {dir_path}.")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="VISCOUS GUBSER FLOW WITH CONSERVED CHARGES (VGCC)",
        description="""Solve the equations of motion for viscous
                                                    Gubser flow with conserved charges.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Developed by the UIUC Nuclear Theory Group.",
    )
    parser.add_argument(
        "--config", default="config.yaml", help="path to config YAML file"
    )
    parser.add_argument(
        "--mode",
        default="initial_condition",
        choices=["initial_condition", "evolution"],
        help="mode to run: initial_condition or evolution",
    )
    parser.add_argument(
        "--eos",
        default="EoS2",
        choices=["EoS1", "EoS2"],
        help="type of equation of state to use",
    )
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the script.
    """
    # Parse command-line arguments.
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Print script header.
    print_header()

    # Load the configuration file.
    logger.debug(f"Loading configuration from {args.config}.")
    try:
        cfg = Config.from_yaml(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration - {e}.")
        sys.exit(1)

    # Setup EoS and EoM parameters.
    eos_params = setup_eos_parameters(cfg)
    eom_params = setup_eom_parameters(cfg)

    # Create the corresponding EoM instance using the selected EoS type.
    try:
        eom_instance = get_eom(args.eos, eos_params=eos_params, eom_params=eom_params)
    except ValueError as exc:
        logger.error(f"EoM are not available - {exc}.")
        sys.exit(1)

    # Setup initial conditions.
    initial_conditions = setup_initial_conditions(cfg)

    # Solve the equations of motion.
    solutions = solve_equations(initial_conditions, eom_instance)

    # Generate output based on the selected mode.
    output_solution(cfg, solutions, eom_instance, mode=args.mode)

    logger.info("Computation complete.")


if __name__ == "__main__":
    main()
