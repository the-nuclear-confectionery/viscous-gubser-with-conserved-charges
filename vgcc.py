#!/usr/bin/env python

import sys
import os
import yaml
import logging
import argparse

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from pathlib import Path
from typing import Union, Dict, Optional, Any

from scipy.interpolate import interp1d
from scipy.integrate import odeint

from utils.constants import print_header, HBARC, tolerance
from utils.analytic_functions import u_x, u_y
from eom.eom_factory import get_eom

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config: dict):
        self.tau_0: float = config.get('tau_0', 1.0)
        self.tau_f: float = config.get('tau_f', 1.6)
        self.tau_step: float = config.get('tau_step', 0.10)
        self.x_min: float = config.get('x_min', -5.0)
        self.x_max: float = config.get('x_max', 5.0)
        self.x_step: float = config.get('x_step', 0.05)
        self.y_min: float = config.get('y_min', -5.0)
        self.y_max: float = config.get('y_max', 5.0)
        self.y_step: float = config.get('y_step', 0.05)
        self.eta_min: float = config.get('eta_min', -0.1)
        self.eta_max: float = config.get('eta_max', 0.1)
        self.step_eta: float = config.get('stepEta', 0.1)
        self.That_0: float = config.get('That_0', 0.05)
        self.pihat_0: float = config.get('pihat_0', 0.0)
        self.muBhat_0: float = config.get('muBhat_0', 0.0025) or 1e-20
        self.muShat_0: float = config.get('muShat_0', 0.000) or 1e-20
        self.muQhat_0: float = config.get('muQhat_0', 0.000) or 1e-20
        self.T_ast: float = config.get('T_ast', 1.000)
        self.muB_ast: float = config.get('muB_ast', 1.00)
        self.muS_ast: float = config.get('muS_ast', 1.00)
        self.muQ_ast: float = config.get('muQ_ast', 1.00)
        self.ETA_OVER_S: float = config.get('eta_over_s', 0.20)
        self.CTAUR: float = config.get('CTAUR', 5.0)
        self.output_dir: str = config.get('output_dir', "data/output/analytic_solutions")

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return cls(config)


def setup_initial_conditions(cfg: Config) -> np.ndarray:
    """Collect de Sitter initial conditions from the configuration file."""
    logger.debug(f"Setting up initial conditions from configuration file.")
    return np.array([cfg.That_0, cfg.muBhat_0, cfg.muShat_0, cfg.muQhat_0, cfg.pihat_0])


def setup_eos_parameters(cfg: Config) -> Dict[str, Union[float, np.ndarray]]:
    """Setup the EoS parameters based on the configuration."""
    logger.debug(f"Setting up EoS parameters from configuration file.")
    eos_params = {
        'T_ast': cfg.T_ast,
        'mu_ast': np.array([cfg.muB_ast, cfg.muS_ast, cfg.muQ_ast]),
    }
    return eos_params

def setup_eom_parameters(cfg: Config) -> Dict[str, float]:
    """Setup the EoM parameters based on the configuration."""
    logger.debug(f"Setting up EoM parameters from configuration file.")
    eom_params = {'CTAUR': cfg.CTAUR, 'ETA_OVER_S': cfg.ETA_OVER_S}
    return eom_params


def solve_equations(initial_conditions: np.ndarray, cfg: Config, kwargs: dict, eom_instance) -> tuple:
    """Solve the equations of motion using the provided eom instance.
       Returns interpolation functions."""
    logger.info(f"Solving equations of motion.")
    # Separate the negative and positive values of rho.
    rhos_neg = np.linspace(-10, 0, 1000)[::-1]
    rhos_pos = np.linspace(0, 10, 1000)
    # Solve the equations of motion for the negative and positive values of rho.
    soln_neg = odeint(eom_instance.eom, initial_conditions, rhos_neg)
    soln_pos = odeint(eom_instance.eom, initial_conditions, rhos_pos)
    # Merge the solutions for negative and positive values of rho.
    t_hat = np.concatenate((soln_neg[:, 0][::-1], soln_pos[:, 0]))
    mu_hat = [np.concatenate((soln_neg[:, i][::-1], soln_pos[:, i])) for i in [1, 2, 3]]
    pi_bar_hat = np.concatenate((soln_neg[:, 4][::-1], soln_pos[:, 4]))
    rhos = np.concatenate((rhos_neg[::-1], rhos_pos))
    # Interpolate the solutions.
    t_interp = interp1d(rhos, t_hat)
    mu_interp = [interp1d(rhos, f) for f in mu_hat]
    pi_interp = interp1d(rhos, pi_bar_hat)
    # Return the interpolation functions.
    logger.info(f"Equations of motion solved.")
    return t_interp, mu_interp, pi_interp


def output_solution(cfg: Config, t_interp: Any, mu_interp: Any, pi_interp: Any,
                            consts: dict, eom_instance: Any, mode: str) -> None:
    """
    Generate output files for either initial conditions (single time step)
    or semi-analytical solutions (multiple time steps). In the case of initial conditions,
    only the first time step (tau = cfg.tau_0) is generated and saved to
    'VGCC_initial_conditions.dat'. Otherwise, each time step is written to its own
    file using the naming convention 'VGCC_tau={tau:.2f}.dat'.
    """
    logger.info(f"Generating ouput for {mode} mode.")

    # Create output directory
    dir_path = Path(cfg.output_dir).absolute()
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Determine the time steps based on the mode.
    if mode == "initial_conditions":
        time_steps = [cfg.tau_0]
    elif mode == "semi-analytical":
        time_steps = np.arange(cfg.tau_0, cfg.tau_f, cfg.tau_step)
    else:
        raise ValueError("Invalid mode. Must be either 'initial_conditions' or 'semi-analytical'.")
    
    # Common formatting for all output files.
    columns = [
        "x", "y", "eta", "energy", "num1", "num2", "num3",
        "u_x", "u_y", "u_eta", "bulk", "pi_xx", "pi_xy",
        "pi_xeta", "pi_yy", "pi_yeta", "pi_etaeta"
    ]
    fmt = {col: (lambda x: f"{x:12.6f}") for col in columns}
    header_str = f'#0 {cfg.x_step} {cfg.y_step} {cfg.step_eta} 0 {cfg.x_min} {cfg.y_min} {cfg.eta_min}\n'
    
    # Use tqdm to display progress bar.
    iterator = tqdm(time_steps, desc="Progress", position=0, leave=False, bar_format='{l_bar}{bar:40} | [Elapsed:{elapsed} | Remaining:{remaining}]', colour='green')
    
    for tau in iterator:
        rows = []
        for x in np.arange(cfg.x_min, cfg.x_max + cfg.x_step, cfg.x_step):
            for y in np.arange(cfg.y_min, cfg.y_max + cfg.y_step, cfg.y_step):
                pis = eom_instance.milne_pi(
                    tau=tau,
                    x=x,
                    y=y,
                    q=1.0,
                    ads_T=t_interp,
                    ads_mu=mu_interp,
                    ads_pi_bar_hat=pi_interp,
                    eos=eom_instance.eos,
                    **consts,
                    tol=tolerance,
                )
                energy = eom_instance.milne_energy(
                    tau=tau,
                    x=x,
                    y=y,
                    q=1.0,
                    ads_T=t_interp,
                    ads_mu=mu_interp,
                    eos=eom_instance.eos,
                    **consts,
                    tol=tolerance,
                )
                nums = eom_instance.milne_number(
                    tau=tau,
                    x=x,
                    y=y,
                    q=1.0,
                    ads_T=t_interp,
                    ads_mu=mu_interp,
                    eos=eom_instance.eos,
                    **consts,
                    tol=tolerance,
                )
                row = [
                    x,
                    y,
                    0,  # eta
                    energy,
                    nums[0],
                    nums[1],
                    nums[2],
                    u_x(tau, x, y, 1.0),
                    u_y(tau, x, y, 1.0),
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
        
        df = pd.DataFrame(rows, columns=columns)
        out_str = df.to_string(index=False, header=False, formatters=fmt)
        
        # Choose the proper file name based on mode.
        if mode == "initial_conditions":
            file_name = "VGCC_initial_conditions.dat"
        else:
            file_name = f'VGCC_tau={tau:.2f}.dat'
        
        file_path = dir_path / file_name
        with open(file_path, "w") as f:
            f.write(header_str)
            f.write(out_str + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Viscous Gubser Flow with Conserved Charges")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--mode", default="initial_conditions",
                        choices=["initial_conditions", "semi-analytical"],
                        help="Mode to run: initial_conditions or semi-analytical")
    parser.add_argument("--eos", default="conformal_plasma",
                        choices=["conformal_plasma", "massless_qgp"],
                        help="Type of equation of state to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    """Main entry point for the script."""
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
        logger.error(f"Failed to load configuration: {e}.")
        sys.exit(1)

    # Read-in the EoS and EoM parameters from the configuration.
    eos_params = setup_eos_parameters(cfg)
    eom_params = setup_eom_parameters(cfg)
    
    # Create the corresponding EoM instance using the selected EoS type.
    try:
        eom_instance = get_eom(args.eos, eos_params=eos_params, eom_params=eom_params)
    except ValueError as exc:
        logger.error(f"EoM for EoS of type {args.eos} are not available: {exc}.")
        sys.exit(1)

    # Setup initial conditions.
    initial_conditions = setup_initial_conditions(cfg)
    
    # Solve the equations of motion.
    t_interp, mu_interp, pi_interp = solve_equations(initial_conditions, cfg, eos_params, eom_instance)
    
    # Generate output based on the selected mode.
    output_solution(cfg, t_interp, mu_interp, pi_interp, eos_params, eom_instance, mode=args.mode)
    
    logger.info("Computation complete.")


if __name__ == "__main__":
    main()