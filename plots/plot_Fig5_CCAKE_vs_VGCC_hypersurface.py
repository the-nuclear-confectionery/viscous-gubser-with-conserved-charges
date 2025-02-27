#!/usr/bin/env python3
# filepath: /Users/jordi/Library/CloudStorage/Box-Box/Research/papers/GUBSER/viscous_gubser_BSQ/plots/plot_Fig5_CCAKE_vs_VGCC_hypersurface.py
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
from typing import List, Union, Tuple

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.patches import FancyArrow
from matplotlib.colors import Normalize

import pandas as pd
import numpy as np
from numpy import sqrt

from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import newton

# Append parent directory for local modules.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.analytic_functions import HBARC, rho
from plots.plotting_settings import customize_axis

from eos.eos_factory import get_eos
from eom.conformal_plasma_eom import ConformalPlasmaEoM

# Create an EOS instance using the conformal plasma EOS.
eos_instance = get_eos("conformal_plasma", temperature_0=0.050, chem_potential_0=[0.050, 0.001, 0.001])
energy = eos_instance.energy

# Create an EoM instance using the same EOS instance and parameters.
eom_instance_eom = ConformalPlasmaEoM(eos_instance, temperature_0=0.050, chem_potential_0=[0.050, 0.001, 0.001])
denergy_drho = eom_instance_eom.denergy_drho

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a figure and axis.
fig, ax_single = plt.subplots(nrows=1, ncols=1, figsize=(3.36, 3.36), dpi=1200, constrained_layout=True)
ax: List[plt.Axes] = [ax_single]

# Centralized style parameters.
SIM_STYLE = {'s': 13.0, 'color': 'red', 'linewidth': 0.0, 'alpha': 1.0}
ANALYTICAL_STYLE = {'ls': '-', 'lw': 2, 'color': 'black'}
ARROW_STYLE_SIM = {'head_width': 0.05, 'head_length': 0.05, 'alpha': 1.0, 'width': 0.01, 'linewidth': 0.5, 'color': 'red'}
ARROW_STYLE_AN = {'head_width': 0.05, 'head_length': 0.05, 'alpha': 1.0, 'width': 0.01, 'linewidth': 0.5, 'color': 'black'}

class Config:
    """Holds configuration parameters."""
    def __init__(self, config: dict) -> None:
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

def read_sim(simulation_path: str) -> pd.DataFrame:
    """
    Reads simulation data from a freeze_out file.
    
    Returns:
        pd.DataFrame: DataFrame containing simulation data.
    """
    simulation_file = os.path.join(simulation_path, 'freeze_out.dat')
    col_names = ['divEener', 'divE_x', 'divE_y', 'gsub', "uout_x", "uout_y",
                 "swsub", "bulksub", "pit_t", "pi_xx", "pi_yy", "pi_zz", "pi_xy",
                 "t", "x", "y", "s", "e", "T", "muB", "muS", "muQ", "w", "cs2"]
    df = pd.read_table(simulation_file, names=col_names, sep=' ', header=0)
    df = df.query('abs(x - y) < 1.E-3').copy()
    df.loc[:, 'r'] = np.sqrt(df['x']**2 + df['y']**2)
    df.loc[:, 'divE_r'] = (df['x'] * df['divE_x'] + df['y'] * df['divE_y']) / df['r']
    logger.info("Simulation data loaded from %s.", simulation_file)
    return df

def find_freezeout_tau(e_interp: interp1d, e_freezeout: float, r: float, q: float) -> float:
    """
    Finds the freeze-out time tau for a given spatial coordinate r using Newton's method.
    """
    try:
        tau = newton(lambda tau: e_freezeout - e_interp(rho(tau, r, q)) / tau ** 4, x0=0.1, x1=0.2)
        return tau
    except RuntimeError as ex:
        logger.error("Newton method failed to converge at r=%s: %s", r, ex)
        raise

def do_freezeout_surfaces(
    rhos_1: np.ndarray,
    rhos_2: np.ndarray,
    xs: np.ndarray,
    T0: float,
    mu0_T0_ratios: np.ndarray,
    pi0: float,
    e_freezeout: float,
    skip_size: int,
    norm_scale: float,
    q: float
) -> Tuple[float, float, float, float, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Computes freeze-out surfaces, associated entropy and normal vectors.

    Returns:
        A tuple: (min_s, max_s, min_tau, max_tau, list_FO_surfaces, list_FO_entropies, list_FO_normals).
    """
    list_FO_surfaces: List[np.ndarray] = []
    list_FO_entropies: List[np.ndarray] = []
    list_FO_normals: List[np.ndarray] = []
    min_tau = 1e99
    max_tau = -1e99

    cfg = Config.from_yaml()  # load default config
    ceos_temp_0 = cfg.ceos_temp_0
    ceos_mu_0 = np.array([cfg.ceos_muB_0, cfg.ceos_muS_0, cfg.ceos_muQ_0])
    consts = {'temperature_0': ceos_temp_0, 'chem_potential_0': ceos_mu_0}  # For legacy use if necessary.

    for k, alpha in enumerate(mu0_T0_ratios):
        y0s = [T0, alpha[0] * T0, alpha[1] * T0, alpha[2] * T0, pi0]
        # Use conformal plasma EoM for numerical integration.
        soln_1 = odeint(eom_instance_eom.eom, y0s, rhos_1)
        soln_2 = odeint(eom_instance_eom.eom, y0s, rhos_2)
        t_hat = np.concatenate((soln_1[:, 0][::-1], soln_2[:, 0]))
        mu_hat = [np.concatenate((soln_1[:, ii][::-1], soln_2[:, ii])) for ii in [1, 2, 3]]
        pi_hat = np.concatenate((soln_1[:, 4][::-1], soln_2[:, 4]))
        rhos = np.concatenate((rhos_1[::-1], rhos_2))

        t_interp = interp1d(rhos, t_hat)
        mu_interp = [interp1d(rhos, f) for f in mu_hat]
        pi_interp = interp1d(rhos, pi_hat)
        # Build energy interpolation along freeze-out.
        e_interp = interp1d(rhos, [energy(*vals) for vals in zip(t_hat, zip(*mu_hat))])
        
        freezeout_times = np.zeros((xs.size, 2))
        for i, x in enumerate(xs):
            try:
                freezeout_times[i] = [x, find_freezeout_tau(e_interp, e_freezeout, x, q)]
            except ValueError:
                freezeout_times[i] = [x, 1.e-12]
        list_FO_surfaces.append(freezeout_times)
        min_tau = min(np.fmin(freezeout_times[:, 1], min_tau))
        max_tau = max(np.fmax(freezeout_times[:, 1], max_tau))

        # Calculate freeze-out normal vectors.
        normal_vectors = np.zeros((xs.size // skip_size, 2))
        for i, var in enumerate(freezeout_times[::skip_size]):
            x, tau_FO = var
            _rho = rho(tau=tau_FO, r=x, q=q)
            try:
                norm_vec = [
                    -denergy_dr( np.array([
                        t_interp(_rho),
                        mu_interp[0](_rho),
                        mu_interp[1](_rho),
                        mu_interp[2](_rho),
                        pi_interp(_rho)
                    ]), tau=tau_FO, r=x, q=q, consts=consts),
                    -denergy_dtau(np.array([
                        t_interp(_rho),
                        mu_interp[0](_rho),
                        mu_interp[1](_rho),
                        mu_interp[2](_rho),
                        pi_interp(_rho)
                    ]), tau=tau_FO, r=x, q=q, consts=consts)
                ]
                norm_val = sqrt(abs(norm_vec[0] ** 2 - norm_vec[1] ** 2))
                normal_vectors[i] = norm_scale * np.array(norm_vec) / norm_val
            except ValueError as err:
                logger.warning("Normal vector calc failed at x=%s, tau_FO=%s, rho=%s: %s", x, tau_FO, _rho, err)
                normal_vectors[i] = [0.0, 0.0]
        list_FO_normals.append(normal_vectors)
    # Here min_s and max_s are placeholders.
    return 0, 0, min_tau, max_tau, list_FO_surfaces, list_FO_entropies, list_FO_normals

def denergy_dtau(
    ys: np.ndarray, tau: float, r: float, q: float, consts: dict
) -> float:
    """
    Compute derivative of energy with respect to tau.
    """
    temperature, mu_B, mu_S, mu_Q, _ = ys
    chem_potential = np.array([mu_B, mu_S, mu_Q])
    derivative = 1 + q ** 2 * (r ** 2 + tau ** 2)
    derivative /= tau * sqrt(1 + q ** 4 * (r ** 2 - tau ** 2) ** 2 + 2 * q ** 2 * (r ** 2 + tau ** 2))
    ret_val = derivative * denergy_drho(ys, rho(tau, r, q)) * tau
    ret_val -= 4.0 * energy(temperature=temperature, chem_potential=chem_potential)
    return ret_val / tau ** 5

def denergy_dr(
    ys: np.ndarray, tau: float, r: float, q: float, consts: dict
) -> float:
    """
    Compute derivative of energy with respect to r.
    """
    derivative = - q * r / tau
    derivative /= sqrt(1 + ((1 + (q * r) ** 2 - (q * tau) ** 2) / (2 * q * tau)) ** 2)
    return derivative * denergy_drho(ys, rho(tau, r, q)) / tau ** 4

def solve_and_plot(
    cfg: Config,
    args: argparse.Namespace,
    fig: plt.Figure,
    ax: List[plt.Axes],
    y0s: np.ndarray,
    mu0_T0_ratios: np.ndarray,
    rhos_1: np.ndarray,
    rhos_2: np.ndarray,
    xs: np.ndarray,
    e_freezeout: float,
    q: float,
    add_labels: bool = True,
    norm_scale: float = 0.1,
    update_color_bar: bool = False,
    plot_s_n: bool = False,
) -> None:
    """
    Compute freeze-out surfaces and plot simulation and analytical data.
    """
    sim_data = read_sim(args.simulation_path)
    xs_sim = np.sort(sim_data['r'].unique())
    skip_size = 1
    _, _, min_tau, max_tau, fo_surfaces, _, fo_normals = do_freezeout_surfaces(
        rhos_1=rhos_1,
        rhos_2=rhos_2,
        xs=xs_sim,
        T0=y0s[0],
        mu0_T0_ratios=mu0_T0_ratios,
        pi0=y0s[2],
        e_freezeout=e_freezeout,
        skip_size=skip_size,
        norm_scale=norm_scale,
        q=q,
    )

    # Plot analytical arrows and surfaces.
    for itr in range(len(mu0_T0_ratios)):
        try:
            freezeout_times = fo_surfaces[itr]
        except IndexError:
            logger.error("Freezeout times index %s out of bounds.", itr)
            raise IndexError("Invalid freezeout index.")
        normal_vectors = fo_normals[itr]
        for i, var in enumerate(freezeout_times[::skip_size]):
            x_val, tau_FO = var
            ax[0].arrow(x=x_val, y=tau_FO,
                        dx=normal_vectors[i, 0],
                        dy=normal_vectors[i, 1],
                        **ARROW_STYLE_AN)
        # Plot analytical freezeout surface.
        ax[0].plot(freezeout_times[:, 0], freezeout_times[:, 1], **ANALYTICAL_STYLE,
                   label='Semi-analytical' if itr == 0 else "")

    # Plot simulation arrows.
    for rec in sim_data.to_dict('records'):
        rec['x'] = rec.pop('r')
        rec['y'] = rec.pop('t')
        rec['dy'] = rec.pop('divEener') * norm_scale
        rec['dx'] = rec.pop('divE_r') * norm_scale
        ax[0].arrow(x=rec['x'], y=rec['y'],
                    dx=rec['dx'], dy=rec['dy'],
                    **ARROW_STYLE_SIM, zorder=10)
    # Plot simulation scatter.
    ax[0].scatter(sim_data['r'], sim_data['t'], **SIM_STYLE, label='CCAKE', zorder=10)
    ax[0].legend(frameon=False, fontsize=14, markerscale=2.0, handlelength=1.5)

    ax[0].text(0.13, 0.3, r'$\mu_0/T_0=0.2$', transform=ax[0].transAxes)
    ax[0].text(0.12, 0.91, r'\textsc{EoS2}', transform=ax[0].transAxes,
        fontsize=10, bbox={'facecolor': 'white', 'boxstyle': 'round', 'linewidth': 0.5},
        horizontalalignment='center',
    )
    return

def beautify(cfg: Config, args: argparse.Namespace) -> None:
    """
    Sets figure styling and kicks off the calculation and plotting.
    """
    fig.patch.set_facecolor('white')
    y0s = np.array([.25 / HBARC, 0.05 / HBARC, 0.0])
    rhos_1 = np.linspace(-10, 0, 1000)[::-1]
    rhos_2 = np.linspace(0, 10, 1000)
    xs = np.linspace(0, 6, 1000)

    solve_and_plot(cfg=cfg, args=args, fig=fig, ax=ax, y0s=y0s,
                   mu0_T0_ratios=np.array([[0.2, 0.0, 0.0]]),
                   rhos_1=rhos_1, rhos_2=rhos_2, xs=xs,
                   e_freezeout=1.0 / HBARC, q=1,
                   add_labels=True, norm_scale=0.10,
                   update_color_bar=True, plot_s_n=True)

    customize_axis(ax=ax[0],
                   x_title=r'$r$ [fm]',
                   y_title=r'$\tau_\mathrm{FO}$ [fm/$c$]',
                   xlim=(-0.05, 2.25),
                   ylim=(0.9, 2.0))
    return

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot Fig5 CCAKE vs VGCC: hypersurface.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--simulation_path", required=True, help="Folder containing the CCAKE results.")
    parser.add_argument("--output_path", required=True, help="Path (excluding filename) to save the figure.")
    return parser.parse_args()

def main() -> None:
    """Main execution function."""
    args = parse_args()
    try:
        cfg = Config.from_yaml(args.config)
    except Exception as e:
        logger.error("Failed to load configuration: %s", e)
        sys.exit(1)
    filename = "Fig5_CCAKE-vs-VGCC_hypersurface.pdf"
    output_path = os.path.join(args.output_path, filename)
    beautify(cfg, args)
    fig.savefig(output_path)
    logger.info("Figure saved to %s", output_path)

if __name__ == '__main__':
    main()