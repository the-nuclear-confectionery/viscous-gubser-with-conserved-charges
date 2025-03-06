#!/usr/bin/env python3
"""
Plot Fig5: CCAKE vs VGCC hypersurface.

This script loads configuration parameters, reads simulation data,
computes freeze‐out surfaces and normal vectors, and plots the hypersurface.
"""

import sys
import os
import logging
import argparse
import yaml
from typing import List, Union, Tuple, Dict, Any, Callable
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import pandas as pd
import numpy as np
from numpy import sqrt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import newton

# Append parent directory for local modules.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import HBARC
from utils.analytic_functions import *
from plots.plotting_settings import customize_axis
from eom.eom_factory import get_eom

# Configure logging.
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global style parameters.
SIM_STYLE = {'s': 13.0, 'color': 'red', 'linewidth': 0.0, 'alpha': 1.0}
ANALYTICAL_STYLE = {'ls': '-', 'lw': 2, 'color': 'black'}
ARROW_STYLE_SIM = {'head_width': 0.05, 'head_length': 0.05,
                   'alpha': 1.0, 'width': 0.01, 'linewidth': 0.5, 'color': 'red'}
ARROW_STYLE_AN = {'head_width': 0.05, 'head_length': 0.05,
                  'alpha': 1.0, 'width': 0.01, 'linewidth': 0.5, 'color': 'black'}


class Config:
    """Configuration object that holds simulation parameters loaded from a YAML file."""
    def __init__(self, config: Dict[str, Any]) -> None:
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
        self.T_hat_0: float = config.get('T_hat_0', 1.20)
        self.pi_bar_hat_0: float = config.get('pi_bar_hat_0', 0.0)
        self.muB_hat_0: float = config.get('muB_hat_0', 1e-20)
        self.muS_hat_0: float = config.get('muS_hat_0', 1e-20)
        self.muQ_hat_0: float = config.get('muQ_hat_0', 1e-20)
        self.T_ast: float = config.get('T_ast', 1.000)
        self.muB_ast: float = config.get('muB_ast', 1.00)
        self.muS_ast: float = config.get('muS_ast', 1.00)
        self.muQ_ast: float = config.get('muQ_ast', 1.00)
        self.ETA_OVER_S: float = config.get('eta_over_s', 0.20)
        self.CTAUR: float = config.get('CTAUR', 5.0)
        self.q: float = config.get('q', 1.0)
        self.output_dir: str = config.get('output_dir', "data/output/analytic_solutions")

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        """Create a Config instance from a YAML configuration file."""
        try:
            with open(path, "r") as file:
                config = yaml.safe_load(file)
            return cls(config)
        except Exception as e:
            logger.error(f"Error loading configuration from {path}: {e}")
            raise


def setup_initial_conditions(cfg: Config) -> np.ndarray:
    """Collect de Sitter initial conditions from the configuration file."""
    logger.debug("Setting up initial conditions from configuration.")
    return np.array([cfg.T_hat_0, cfg.muB_hat_0, cfg.muS_hat_0,
                     cfg.muQ_hat_0, cfg.pi_bar_hat_0])


def setup_eos_parameters(cfg: Config) -> Dict[str, Union[float, np.ndarray]]:
    """Setup the Equation of State (EoS) parameters based on the configuration."""
    logger.debug("Setting up EoS parameters from configuration.")
    return {'T_ast': cfg.T_ast,
            'mu_ast': np.array([cfg.muB_ast, cfg.muS_ast, cfg.muQ_ast])}


def setup_eom_parameters(cfg: Config) -> Dict[str, float]:
    """Setup the equations of motion (EoM) parameters based on the configuration."""
    logger.debug("Setting up EoM parameters from configuration.")
    return {'CTAUR': cfg.CTAUR, 'ETA_OVER_S': cfg.ETA_OVER_S}


def solve_equations(initial_conditions: np.ndarray,
                    eom_instance: Any
                    ) -> Tuple[Callable, List[Callable], Callable, Callable]:
    """
    Solve the equations of motion using the provided EoM instance.
    Returns interpolation functions for temperature (T), chemical potentials (mu),
    and shear (pi), as well as energy.
    """
    logger.info("Solving equations of motion.")
    # Define rho ranges.
    rhos_neg = np.linspace(-10, 0, 1000)[::-1]
    rhos_pos = np.linspace(0, 10, 1000)
    soln_neg = odeint(eom_instance.eom, initial_conditions, rhos_neg)
    soln_pos = odeint(eom_instance.eom, initial_conditions, rhos_pos)
    # Merge solutions.
    T_hat = np.concatenate((soln_neg[:, 0][::-1], soln_pos[:, 0]))
    mu_hat = [np.concatenate((soln_neg[:, i][::-1], soln_pos[:, i]))
              for i in [1, 2, 3]]
    pi_bar_hat = np.concatenate((soln_neg[:, 4][::-1], soln_pos[:, 4]))
    rhos = np.concatenate((rhos_neg[::-1], rhos_pos))
    # Interpolation functions.
    T_interp = interp1d(rhos, T_hat)
    mu_interp = [interp1d(rhos, m) for m in mu_hat]
    pi_interp = interp1d(rhos, pi_bar_hat)
    # Energy interpolation.
    T_vals = T_interp(rhos)
    mu_vals = np.array([f(rhos) for f in mu_interp])
    energy_vals = [eom_instance.eos.energy(T_vals[i], mu_vals[:, i])
                   for i in range(len(rhos))]
    e_interp = interp1d(rhos, energy_vals)
    logger.info("Equations of motion solved.")
    return T_interp, mu_interp, pi_interp, e_interp


def read_sim(simulation_path: str) -> pd.DataFrame:
    """
    Reads simulation data from a freeze_out file.

    Returns:
        pd.DataFrame: DataFrame containing simulation data.
    """
    simulation_file = Path(simulation_path) / "freeze_out.dat"
    col_names = ['divEener', 'divE_x', 'divE_y', 'gsub', "uout_x", "uout_y",
                 "swsub", "bulksub", "pit_t", "pi_xx", "pi_yy", "pi_zz",
                 "pi_xy", "t", "x", "y", "s", "e", "T", "muB", "muS", "muQ",
                 "w", "cs2"]
    df = pd.read_table(simulation_file, names=col_names, sep=' ', header=0)
    # Filter data and calculate additional columns.
    df = df.query('abs(x - y) < 1e-3').copy()
    df['r'] = np.sqrt(df['x'] ** 2 + df['y'] ** 2)
    df['divE_r'] = (df['x'] * df['divE_x'] +
                    df['y'] * df['divE_y']) / df['r']
    logger.info(f"Simulation data loaded from {simulation_file}.")
    return df


def find_freezeout_tau(e_interp: Callable, e_freezeout: float, r: float, q: float) -> float:
    """
    Finds the freeze-out time tau for a given spatial coordinate r using Newton's method.
    """
    try:
        tau = newton(lambda tau: e_freezeout - e_interp(rho(tau, r, q)) / tau**4,
                     x0=0.1, x1=0.2)
        return tau
    except RuntimeError as ex:
        logger.error(f"Newton method failed to converge at r={r}: {ex}")
        raise


def do_freezeout_surfaces(
    xs: np.ndarray,
    solutions: Tuple[Callable, List[Callable], Callable, Callable],
    eom_instance: Any,
    initial_conditions: np.ndarray,
    muB_hat_0_over_T_hat_0_ratio: float,
    q: float,
    e_freezeout: float,
    skip_size: int,
    norm_scale: float,
) -> Tuple[float, float, float, float, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Computes freeze-out surfaces and associated normal vectors.

    Returns:
        Tuple containing placeholders for min_s/max_s, min/max tau values,
        and lists for freeze-out surfaces, entropies (placeholder), and normals.
    """
    FO_surfaces: List[np.ndarray] = []
    FO_entropies: List[np.ndarray] = []  # Placeholder for future entropy calculations.
    FO_normals: List[np.ndarray] = []
    min_tau = float('inf')
    max_tau = float('-inf')

    # Unpack interpolation functions.
    T_interp, mu_interp, pi_interp, e_interp = solutions

    # Compute freeze-out times.
    freezeout_times = np.zeros((len(xs), 2))
    for i, x in enumerate(xs):
        try:
            freezeout_times[i] = [x, find_freezeout_tau(e_interp, e_freezeout, x, q)]
        except Exception:
            freezeout_times[i] = [x, 1.e-12]
    FO_surfaces.append(freezeout_times)
    min_tau = min(np.min(freezeout_times[:, 1]), min_tau)
    max_tau = max(np.max(freezeout_times[:, 1]), max_tau)

    # Calculate freeze-out normal vectors.
    normals = np.zeros((len(xs) // skip_size, 2))
    for i, (x, tau_FO) in enumerate(freezeout_times[::skip_size]):
        _rho = rho(tau_FO, x, q)
        try:
            state = np.array([
                T_interp(_rho),
                mu_interp[0](_rho),
                mu_interp[1](_rho),
                mu_interp[2](_rho),
                pi_interp(_rho)
            ])
            dE_dr = denergy_dr(state, tau=tau_FO, r=x, q=q, eom_instance=eom_instance)
            dE_dtau = denergy_dtau(state, tau=tau_FO, r=x, q=q, eom_instance=eom_instance)
            norm_vec = [-dE_dr, -dE_dtau]
            norm_val = np.sqrt(abs(norm_vec[0] ** 2 - norm_vec[1] ** 2))
            normals[i] = norm_scale * np.array(norm_vec) / norm_val
        except Exception as err:
            logger.warning(f"Normal vector calc failed at x={x}, tau_FO={tau_FO}, rho={_rho}: {err}")
            normals[i] = [0.0, 0.0]
    FO_normals.append(normals)

    return 0, 0, min_tau, max_tau, FO_surfaces, FO_entropies, FO_normals


def denergy_dtau(ys: np.ndarray, tau: float, r: float, q: float,
                 eom_instance: Any) -> float:
    """
    Compute derivative of energy with respect to tau.
    """
    temperature, mu_B, mu_S, mu_Q, _ = ys
    chem_potential = np.array([mu_B, mu_S, mu_Q])
    derivative = 1 + q**2 * (r**2 + tau**2)
    derivative /= tau * sqrt(1 + q**4 * (r**2 - tau**2)**2 + 2 * q**2 * (r**2 + tau**2))
    ret_val = derivative * eom_instance.denergy_drho(ys, rho(tau, r, q)) * tau
    ret_val -= 4.0 * eom_instance.eos.energy(temperature, chem_potential)
    return ret_val / tau**5


def denergy_dr(ys: np.ndarray, tau: float, r: float, q: float,
               eom_instance: Any) -> float:
    """
    Compute derivative of energy with respect to r.
    """
    derivative = -q * r / tau
    derivative /= sqrt(1 + ((1 + (q * r)**2 - (q * tau)**2) / (2 * q * tau))**2)
    return derivative * eom_instance.denergy_drho(ys, rho(tau, r, q)) / tau**4


def solve_and_plot(
    args: argparse.Namespace,
    cfg: Config,
    ax: plt.Axes,
    solutions: Tuple[Callable, List[Callable], Callable, Callable],
    eom_instance: Any,
    initial_conditions: np.ndarray,
    e_freezeout: float,
    norm_scale: float = 0.1
) -> None:
    """
    Compute freeze-out surfaces and plot simulation and analytical data.
    """
    sim_data = read_sim(args.simulation_path)
    xs_sim = np.sort(sim_data['r'].unique())
    skip_size = 1
    mu_ratio = cfg.muB_hat_0 / cfg.T_hat_0

    _, _, min_tau, max_tau, FO_surfaces, _, FO_normals = do_freezeout_surfaces(
        xs=xs_sim,
        solutions=solutions,
        eom_instance=eom_instance,
        initial_conditions=initial_conditions,
        muB_hat_0_over_T_hat_0_ratio=mu_ratio,
        q=cfg.q,
        e_freezeout=e_freezeout,
        skip_size=skip_size,
        norm_scale=norm_scale,
    )

    freezeout_times = FO_surfaces[0]
    normals = FO_normals[0]
    # Plot analytical arrows.
    for i, (x_val, tau_FO) in enumerate(freezeout_times[::skip_size]):
        ax.arrow(x=x_val, y=tau_FO,
                 dx=normals[i, 0],
                 dy=normals[i, 1],
                 **ARROW_STYLE_AN)
    # Plot analytical freeze-out surface.
    ax.plot(freezeout_times[:, 0],
            freezeout_times[:, 1],
            **ANALYTICAL_STYLE,
            label='Semi-analytical')

    # Plot simulation arrows.
    for rec in sim_data.to_dict('records'):
        rec['x'] = rec.pop('r')
        rec['y'] = rec.pop('t')
        rec['dx'] = rec.pop('divE_r') * norm_scale
        rec['dy'] = rec.pop('divEener') * norm_scale
        ax.arrow(x=rec['x'], y=rec['y'],
                 dx=rec['dx'], dy=rec['dy'],
                 **ARROW_STYLE_SIM, zorder=10)
    # Scatter plot simulation data.
    ax.scatter(sim_data['r'], sim_data['t'], **SIM_STYLE,
               label='CCAKE', zorder=10)
    ax.legend(frameon=False, fontsize=14, markerscale=2.0, handlelength=1.5, loc='lower left')
    ax.text(0.13, 0.3, r'$\hat{\mu}_{Y,0}/\hat{T}_0=$' + f' {mu_ratio:.2f}', transform=ax.transAxes)
    ax.text(0.12, 0.91, r'\textsc{EoS2}', transform=ax.transAxes,
            fontsize=10,
            bbox={'facecolor': 'white', 'boxstyle': 'round', 'linewidth': 0.5},
            horizontalalignment='center')
    return


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot Fig5 CCAKE vs VGCC: hypersurface."
    )
    parser.add_argument("--config",
                        default="config.yaml",
                        help="Path to the config YAML file")
    parser.add_argument("--eos",
                        default="EoS2",
                        choices=["EoS1", "EoS2"],
                        help="Type of equation of state to use")
    parser.add_argument("--simulation_path",
                        required=True,
                        help="Folder containing the CCAKE results.")
    parser.add_argument("--output_path",
                        required=True,
                        help="Path (excluding filename) to save the figure.")
    parser.add_argument("--debug",
                        action="store_true", 
                        help="enable debug logging")
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    try:
        cfg = Config.from_yaml(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    output_dir = Path(args.output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    eos_params = setup_eos_parameters(cfg)
    eom_params = setup_eom_parameters(cfg)
    try:
        eom_instance = get_eom(args.eos,
                               eos_params=eos_params,
                               eom_params=eom_params)
    except ValueError as exc:
        logger.error(f"EoM are not available - {exc}.")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(3.36, 3.36),
                           dpi=1200,
                           constrained_layout=True)
    fig.patch.set_facecolor('white')

    initial_conditions = setup_initial_conditions(cfg)
    solutions = solve_equations(initial_conditions, eom_instance)
    solve_and_plot(args, cfg, ax, solutions, eom_instance,
                   initial_conditions,
                   e_freezeout=1.0 / HBARC,
                   norm_scale=0.10)

    customize_axis(ax=ax,
                   x_title=r'$r$ [fm]',
                   y_title=r'$\tau_\mathrm{FO}$ [fm/$c$]',
                   xlim=(-0.1, 2.4),
                   ylim=(0.9, 1.9))

    output_file = output_dir / "Fig5_CCAKE-vs-VGCC_hypersurface.pdf"
    fig.savefig(output_file)
    logger.info(f"Figure saved to {output_file}")


if __name__ == '__main__':
    main()
