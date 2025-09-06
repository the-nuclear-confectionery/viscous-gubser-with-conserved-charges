import sys
import logging
import argparse
import yaml
from typing import List, Union, Tuple, Dict, Any, Callable
from pathlib import Path

from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import newton

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap
from matplotlib.cm import ScalarMappable


from matplotlib.patches import FancyArrow

from matplotlib.colors import Normalize
from matplotlib import cm

# Append parent directory for local modules.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from plots.plotting_settings import customize_axis
from utils.constants import HBARC
from utils.analytic_functions import rho, milne_T, milne_mu, milne_entropy
from eom.eom_factory import get_eom

# Configure logging.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration object that holds simulation parameters loaded from a YAML file."""

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
    return np.array(
        [cfg.T_hat_0, cfg.muB_hat_0, cfg.muS_hat_0, cfg.muQ_hat_0, cfg.pi_bar_hat_0]
    )


def setup_eos_parameters(cfg: Config) -> Dict[str, Union[float, np.ndarray]]:
    """Setup the Equation of State (EoS) parameters based on the configuration."""
    logger.debug("Setting up EoS parameters from configuration.")
    return {
        "T_ast": cfg.T_ast,
        "mu_ast": np.array([cfg.muB_ast, cfg.muS_ast, cfg.muQ_ast]),
    }


def setup_eom_parameters(cfg: Config) -> Dict[str, float]:
    """Setup the equations of motion (EoM) parameters based on the configuration."""
    logger.debug("Setting up EoM parameters from configuration.")
    return {"CTAUR": cfg.CTAUR, "ETA_OVER_S": cfg.ETA_OVER_S}


def solve_equations(
    initial_conditions: np.ndarray, eom_instance: Any, ideal_evolution: bool = False
) -> Tuple[Callable, List[Callable], Callable, Callable]:
    """
    Solve the equations of motion using the provided EoM instance.
    Returns interpolation functions for temperature (T), chemical potentials (mu),
    and shear (pi), as well as energy.
    """
    logger.info("Solving equations of motion.")
    # Define rho ranges.
    rhos_neg = np.linspace(-12, 0, 1000)[::-1]
    rhos_pos = np.linspace(0, 12, 1000)
    soln_neg = odeint(
        eom_instance.eom, initial_conditions, rhos_neg, args=(ideal_evolution,)
    )
    soln_pos = odeint(
        eom_instance.eom, initial_conditions, rhos_pos, args=(ideal_evolution,)
    )
    # Merge solutions.
    T_hat = np.concatenate((soln_neg[:, 0][::-1], soln_pos[:, 0]))
    mu_hat = [np.concatenate((soln_neg[:, i][::-1], soln_pos[:, i])) for i in [1, 2, 3]]
    pi_bar_hat = np.concatenate((soln_neg[:, 4][::-1], soln_pos[:, 4]))
    rhos = np.concatenate((rhos_neg[::-1], rhos_pos))
    # Interpolation functions.
    T_interp = interp1d(rhos, T_hat)
    mu_interp = [interp1d(rhos, m) for m in mu_hat]
    pi_interp = interp1d(rhos, pi_bar_hat)
    # Energy interpolation.
    T_vals = T_interp(rhos)
    mu_vals = np.array([f(rhos) for f in mu_interp])
    energy_vals = [
        eom_instance.eos.energy(T_vals[i], mu_vals[:, i]) for i in range(len(rhos))
    ]
    e_interp = interp1d(rhos, energy_vals)
    logger.info("Equations of motion solved.")
    return T_interp, mu_interp, pi_interp, e_interp


def find_freezeout_tau(
    e_interp: interp1d,
    e_freezeout: float,
    r: float,
    q: float,
) -> float:
    def f(tau: float) -> float:
        return e_freezeout - e_interp(rho(tau, r, q)) / tau**4

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
    return newton(lambda t: 4.0 / np.tanh(mu / t) - mu / t - s_n, 0.1)


def denergy_dtau(
    ys: np.ndarray, tau: float, r: float, q: float, eom_instance: Any
) -> float:
    """
    Compute derivative of energy with respect to tau.
    """
    temperature, mu_B, mu_S, mu_Q, _ = ys
    chem_potential = np.array([mu_B, mu_S, mu_Q])
    derivative = 1 + q**2 * (r**2 + tau**2)
    derivative /= tau * np.sqrt(
        1 + q**4 * (r**2 - tau**2) ** 2 + 2 * q**2 * (r**2 + tau**2)
    )
    ret_val = derivative * eom_instance.denergy_drho(ys, rho(tau, r, q)) * tau
    ret_val -= 4.0 * eom_instance.eos.energy(temperature, chem_potential)
    return ret_val / tau**5


def denergy_dr(
    ys: np.ndarray, tau: float, r: float, q: float, eom_instance: Any
) -> float:
    """
    Compute derivative of energy with respect to r.
    """
    derivative = -q * r / tau
    derivative /= np.sqrt(
        1 + ((1 + (q * r) ** 2 - (q * tau) ** 2) / (2 * q * tau)) ** 2
    )
    return derivative * eom_instance.denergy_drho(ys, rho(tau, r, q)) / tau**4


def find_color_indices(
    min_s: float,
    max_s: float,
    num_colors: int,
    s: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    ds = (max_s - min_s) / num_colors
    result = np.floor((s - min_s) / ds)
    return result


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
) -> Tuple[
    float, float, float, float, List[np.ndarray], List[np.ndarray], List[np.ndarray]
]:
    """
    Computes freeze-out surfaces and associated normal vectors.

    Returns:
        Tuple containing placeholders for min_s/max_s, min/max tau values,
        and lists for freeze-out surfaces, entropies (placeholder), and normals.
    """
    FO_surfaces: List[np.ndarray] = []
    FO_entropies: List[np.ndarray] = []  # Placeholder for future entropy calculations.
    FO_normals: List[np.ndarray] = []
    min_s = +1e99
    max_s = -1e99
    min_tau = +1e99
    max_tau = -1e99

    # Unpack interpolation functions.
    T_interp, mu_interp, pi_interp, e_interp = solutions

    # Compute freeze-out times.
    freezeout_times = np.zeros((len(xs), 2))
    for i, x in enumerate(xs):
        try:
            freezeout_times[i] = [x, find_freezeout_tau(e_interp, e_freezeout, x, q)]
        except Exception:
            freezeout_times[i] = [x, 1.0e-12]
    FO_surfaces.append(freezeout_times)
    min_tau = min(np.fmin(freezeout_times[:, 1], min_tau))
    max_tau = max(np.fmax(freezeout_times[:, 1], max_tau))

    for i, (x, tau_FO) in enumerate(freezeout_times):
        _rho = rho(tau_FO, x, q)
        try:
            freezeout_s = milne_entropy(
                tau=tau_FO,
                x=x,
                y=0.0,
                q=q,
                interpolated_T_hat=T_interp,
                interpolated_mu_hat=mu_interp,
                eos_instance=eom_instance.eos,
            )
            FO_entropies.append(freezeout_s)
            min_s = min(np.min(freezeout_s), min_s)
            max_s = max(np.max(freezeout_s), max_s)
        except ValueError:
            FO_entropies.append(0.0)

    # Calculate freeze-out normal vectors.
    normals = np.zeros((len(xs) // skip_size, 2))
    for i, (x, tau_FO) in enumerate(freezeout_times[::skip_size]):
        _rho = rho(tau_FO, x, q)
        try:
            state = np.array(
                [
                    T_interp(_rho),
                    mu_interp[0](_rho),
                    mu_interp[1](_rho),
                    mu_interp[2](_rho),
                    pi_interp(_rho),
                ]
            )
            dE_dr = denergy_dr(state, tau=tau_FO, r=x, q=q, eom_instance=eom_instance)
            dE_dtau = denergy_dtau(
                state, tau=tau_FO, r=x, q=q, eom_instance=eom_instance
            )
            norm_vec = [-dE_dr, -dE_dtau]
            norm_val = np.sqrt(abs(norm_vec[0] ** 2 - norm_vec[1] ** 2))
            normals[i] = norm_scale * np.array(norm_vec) / norm_val
        except Exception as err:
            logger.warning(
                f"Normal vector calc failed at x={x}, tau_FO={tau_FO}, rho={_rho}: {err}"
            )
            normals[i] = [0.0, 0.0]
    FO_normals.append(normals)

    return min_s, max_s, min_tau, max_tau, FO_surfaces, FO_entropies, FO_normals


def dx_dtau(
    ys: np.ndarray, tau: Union[float, np.ndarray], q: float
) -> Union[float, np.ndarray]:
    x, y, eta = ys
    r2 = x**2 + y**2
    v_r = 2 * q * tau / (1 + q**2 * (r2 + tau**2))
    r = np.sqrt(r2)
    return np.array([x / r, y / y, 0]) * v_r


def plot_hypersurface(
    args: argparse.Namespace,
    cfg: Config,
    fig: plt.Figure,
    ax: plt.Axes,
    solutions: tuple,
    eom_instance: Any,
    initial_conditions: np.ndarray,
    e_freezeout: float,
    norm_scale: float,
    draw_colorbar: bool = False,
) -> int:
    """
    Plot the hypersurface of the freeze-out surface.
    """
    # Define the range of x values.
    xs = np.linspace(0, 6, 1000)
    muB_hat_0_over_T_hat_0_ratio = initial_conditions[1] / initial_conditions[0]

    skip_size = 8
    min_s, max_s, min_tau, max_tau, FO_surfaces, FO_entropies, FO_normals = (
        do_freezeout_surfaces(
            xs=xs,
            solutions=solutions,
            eom_instance=eom_instance,
            initial_conditions=initial_conditions,
            muB_hat_0_over_T_hat_0_ratio=muB_hat_0_over_T_hat_0_ratio,
            q=cfg.q,
            e_freezeout=e_freezeout,
            skip_size=skip_size,
            norm_scale=norm_scale,
        )
    )

    freezeout_times = FO_surfaces[0]
    freezeout_s = np.array(FO_entropies)
    normal_vectors = FO_normals[0]

    arrows = np.zeros((xs.size // skip_size,), dtype=FancyArrow)
    for i, var in enumerate(freezeout_times[::skip_size]):
        x, tau_FO = var
        arrows[i] = ax.arrow(
            x=x,
            y=tau_FO,
            dx=normal_vectors[i, 0],
            dy=normal_vectors[i, 1],
            head_width=0.02,
            head_length=0.01,
            lw=0.5,
            color="black",
        )

    heat_map = get_cmap("copper", freezeout_s.size)

    ax.scatter(
        freezeout_times[::, 0],
        freezeout_times[::, 1],
        c=find_color_indices(
            min_s=2,
            max_s=9,
            num_colors=freezeout_s.size,
            s=freezeout_s,
        ),
        s=1.00,
        cmap=heat_map,
        norm=Normalize(vmin=0, vmax=freezeout_s.size),
    )

    if draw_colorbar:
        norm = Normalize(vmin=4.5, vmax=8.5)
        s = ScalarMappable(norm=norm, cmap=heat_map)
        cax = fig.colorbar(s, ax=ax, orientation="vertical", pad=0.02, format="%.2f").ax
        cax.yaxis.set_ticks(np.linspace(min_s, max_s, 7))
        for t in cax.get_yticklabels():
            t.set_fontsize(10)
        cax.set_ylabel(r"$\displaystyle s(\tau, x)$ [GeV$^{3}$]", fontsize=12)

    return 0


def plot_trajectories(
    args: argparse.Namespace,
    cfg: Config,
    fig: plt.Figure,
    ax: plt.Axes,
    solutions: tuple,
    eom_instance: Any,
    initial_conditions: np.ndarray,
    e_freezeout: float,
) -> int:
    evol_taus_log = np.linspace(np.log(0.05), np.log(10), 10000)
    evol_taus = np.exp(evol_taus_log)

    T_interp, mu_interp, pi_interp, e_interp = solutions

    # Plot trajectories
    rs = np.linspace(0.001, 5.001, 10000)
    evol_mus = np.zeros((rs.size, evol_taus.size))
    evol_temps = np.zeros_like(evol_mus)
    for nn, r0 in enumerate(rs):
        evol_xs = odeint(
            dx_dtau, np.array([r0, r0, 0]), np.exp(evol_taus_log), args=(1.0,)
        )
        evol_rs = np.sqrt(evol_xs[:, 0] ** 2 + evol_xs[:, 1] ** 2)
        evol_mus[nn] = milne_mu(evol_taus, evol_rs, 1.0, mu_interp[0])
        evol_temps[nn] = milne_T(evol_taus, evol_rs, 1.0, T_interp)

    _, _, _, color_mesh = ax.hist2d(
        evol_mus.reshape(
            -1,
        ),
        evol_temps.reshape(
            -1,
        ),
        bins=200,
        # cmap=[viridis, plasma][itr - 1],
        norm="log",
        alpha=1.0,
        density=True,
    )

    # cax_2 = fig.colorbar(color_mesh, ax=ax, orientation='vertical',
    #                         pad=0.02, format='%.2f').ax
    # for t in cax_2.get_yticklabels():
    #     t.set_fontsize(10)
    # cax_2.set_ylabel(r'count (normalized)', fontsize=12)

    return 0


def plot_ideal_trajectories(
    args: argparse.Namespace,
    cfg: Config,
    ax: plt.Axes,
    fig: plt.Figure,
    solutions: tuple,
    eom_instance: Any,
    initial_conditions: np.ndarray,
    e_freezeout: float,
) -> int:
    evol_taus_log = np.linspace(np.log(0.001), np.log(10), 1000)
    evol_taus = np.exp(evol_taus_log)

    T_interp, mu_interp, pi_interp, e_interp = solutions

    # Plot ideal trajectories
    rs = np.linspace(0.01, 5.01, 1000)
    evol_mus = np.zeros((rs.size, evol_taus.size))
    evol_temps = np.zeros_like(evol_mus)
    for nn, r0 in enumerate([1e-5]):
        evol_xs = odeint(
            dx_dtau, np.array([r0, r0, 0]), np.exp(evol_taus_log), args=(1.0,)
        )
        evol_rs = np.sqrt(evol_xs[:, 0] ** 2 + evol_xs[:, 1] ** 2)
        evol_mus = milne_mu(evol_taus, evol_rs, 1.0, mu_interp[0])
        evol_temps = milne_T(evol_taus, evol_rs, 1.0, T_interp)

        ax.plot(evol_mus, evol_temps, lw=1, color="black", ls="dashed")

    return 0


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot Fig3 VGCC freezeout and trajectories."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the config YAML file"
    )
    parser.add_argument(
        "--eos",
        default="EoS1",
        choices=["EoS1", "EoS2"],
        help="Type of equation of state to use",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path (excluding filename) to save the figure.",
    )
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    return parser.parse_args()


def main():
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
        eom_instance = get_eom(args.eos, eos_params=eos_params, eom_params=eom_params)
    except ValueError as exc:
        logger.error(f"EoM are not available - {exc}.")
        sys.exit(1)
    fig, ax = plt.subplots(
        nrows=1, ncols=3, figsize=(11.1, 3.36), dpi=1200, width_ratios=[1.15, 1, 1]
    )

    e_freezeout = 1 / HBARC
    norm_scale = 0.075

    # Plot hypersurface for mu/T = 0
    initial_conditions = np.array([cfg.T_hat_0, 0 * cfg.T_hat_0, 0, 0, 0])
    solutions = solve_equations(initial_conditions, eom_instance)
    plot_hypersurface(
        args,
        cfg,
        fig,
        ax[0],
        solutions,
        eom_instance,
        initial_conditions,
        e_freezeout,
        norm_scale,
    )

    # Plot hypersurface for mu/T = 10
    initial_conditions = np.array([cfg.T_hat_0, 10 * cfg.T_hat_0, 0, 0, 0])
    solutions = solve_equations(initial_conditions, eom_instance)
    plot_hypersurface(
        args,
        cfg,
        fig,
        ax[0],
        solutions,
        eom_instance,
        initial_conditions,
        e_freezeout,
        norm_scale,
    )

    # Plot hypersurface for mu/T = 15
    initial_conditions = np.array([cfg.T_hat_0, 15 * cfg.T_hat_0, 0, 0, 0])
    solutions = solve_equations(initial_conditions, eom_instance)
    plot_hypersurface(
        args,
        cfg,
        fig,
        ax[0],
        solutions,
        eom_instance,
        initial_conditions,
        e_freezeout,
        norm_scale,
        draw_colorbar=True,
    )

    # Plot trajectories for mu/T = 10
    initial_conditions = np.array([cfg.T_hat_0, 10 * cfg.T_hat_0, 0, 0, 0])
    solutions = solve_equations(initial_conditions, eom_instance)
    # plot_trajectories(args, cfg, fig, ax[1], solutions, eom_instance, initial_conditions, e_freezeout)
    ideal_solutions = solve_equations(
        initial_conditions, eom_instance, ideal_evolution=True
    )
    plot_ideal_trajectories(
        args,
        cfg,
        ax[1],
        fig,
        ideal_solutions,
        eom_instance,
        initial_conditions,
        e_freezeout,
    )

    mu_Y = np.linspace(0, 4, 1000)
    T = np.linspace(0, 0.6, 1000)
    MU_Y, T = np.meshgrid(mu_Y, T)
    ax[1].text(0.35, 0.07, r"FO", fontsize=10, color="white", ha="center", zorder=13)
    ax[1].text(
        0.35, 0.05, r"region", fontsize=10, color="white", ha="center", zorder=13
    )
    ax[1].contourf(
        MU_Y,
        T,
        eom_instance.eos.energy(T, MU_Y) / HBARC**3,
        levels=[0, 1],
        cmap=cm.PuBu_r,
        alpha=0.85,
        zorder=11,
    )
    ax[1].contour(
        MU_Y,
        T,
        eom_instance.eos.energy(T, MU_Y) / HBARC**3,
        levels=[0, 1],
        colors="deepskyblue",
        linestyles="solid",
        linewidths=1.5,
        zorder=12,
    )
    ax[2].text(0.35, 0.07, r"FO", fontsize=10, color="white", ha="center", zorder=13)
    ax[2].text(
        0.35, 0.05, r"region", fontsize=10, color="white", ha="center", zorder=13
    )
    ax[2].contourf(
        MU_Y,
        T,
        eom_instance.eos.energy(T, MU_Y) / HBARC**3,
        levels=[0, 1],
        cmap=cm.PuBu_r,
        alpha=0.85,
        zorder=11,
    )
    ax[2].contour(
        MU_Y,
        T,
        eom_instance.eos.energy(T, MU_Y) / HBARC**3,
        levels=[0, 1],
        colors="deepskyblue",
        linestyles="solid",
        linewidths=1.5,
        zorder=12,
    )

    xs = np.linspace(0, 40, 1000)
    mus = milne_mu(1.0, xs, 1.0, solutions[1][0])
    Ts = milne_T(1.0, xs, 1.0, solutions[0])
    ax[1].text(
        mus[0] + 0.05, Ts[0] - 0.02, r"$\tau_0$", fontsize=10, color="blue", ha="left"
    )
    ax[1].plot(mus, Ts, lw=1, color="blue", ls="dotted", zorder=8)
    mus = milne_mu(1.2, xs, 1.0, solutions[1][0])
    Ts = milne_T(1.2, xs, 1.0, solutions[0])
    ax[1].text(
        mus[0] + 0.05, Ts[0] - 0.02, r"$\tau_1$", fontsize=10, color="blue", ha="left"
    )
    ax[1].plot(mus, Ts, lw=1, color="blue", ls="dotted", zorder=8)
    mus = milne_mu(2.0, xs, 1.0, solutions[1][0])
    Ts = milne_T(2.0, xs, 1.0, solutions[0])
    ax[1].text(
        mus[0] + 0.1, Ts[0] - 0.02, r"$\tau_2$", fontsize=10, color="blue", ha="left"
    )
    ax[1].plot(mus, Ts, lw=1, color="blue", ls="dotted", zorder=8)

    taus = np.linspace(1.0, 15, 1000)
    mus = milne_mu(taus, 0.0, 1.0, solutions[1][0])
    Ts = milne_T(taus, 0.0, 1.0, solutions[0])
    ax[1].text(mus[0], Ts[0] + 0.02, r"$x_0$", fontsize=10, color="r", ha="center")
    ax[1].scatter(mus[0], Ts[0], color="r", edgecolors="b", lw=0.5, s=10, zorder=10)
    ax[1].plot(mus, Ts, lw=1, color="red", ls="solid", zorder=9)
    mus = milne_mu(taus, 1.5, 1.0, solutions[1][0])
    Ts = milne_T(taus, 1.5, 1.0, solutions[0])
    ax[1].scatter(mus[0], Ts[0], color="r", edgecolors="b", lw=0.5, s=10, zorder=10)
    ax[1].text(mus[0], Ts[0] + 0.015, r"$x_1$", fontsize=10, color="r", ha="center")
    ax[1].plot(mus, Ts, lw=1, color="red", ls="solid", zorder=9)
    mus = milne_mu(taus, 2.0, 1.0, solutions[1][0])
    Ts = milne_T(taus, 2.0, 1.0, solutions[0])
    ax[1].scatter(mus[0], Ts[0], color="r", edgecolors="b", lw=0.5, s=10, zorder=10)
    ax[1].text(mus[0], Ts[0] + 0.015, r"$x_2$", fontsize=10, color="r", ha="center")
    ax[1].plot(mus, Ts, lw=1, color="red", ls="solid", zorder=9)
    mus = milne_mu(taus, 3.0, 1.0, solutions[1][0])
    Ts = milne_T(taus, 3.0, 1.0, solutions[0])
    ax[1].scatter(mus[0], Ts[0], color="r", edgecolors="b", lw=0.5, s=10, zorder=10)
    ax[1].text(
        mus[0] - 0.05, Ts[0] + 0.02, r"$x_3$", fontsize=10, color="r", ha="center"
    )
    ax[1].plot(mus, Ts, lw=1, color="red", ls="solid", zorder=9)
    mus = milne_mu(taus, 4.0, 1.0, solutions[1][0])
    Ts = milne_T(taus, 4.0, 1.0, solutions[0])
    ax[1].scatter(mus[0], Ts[0], color="r", edgecolors="b", lw=0.5, s=10, zorder=10)
    ax[1].plot(mus, Ts, lw=1, color="red", ls="solid", zorder=9)

    ax[1].text(
        3.9, 0.13, r"$x_0=0.0\;\mathrm{fm}$", fontsize=10, color="r", ha="right"
    )
    ax[1].text(
        3.9, 0.10, r"$x_1=1.5\;\mathrm{fm}$", fontsize=10, color="r", ha="right"
    )
    ax[1].text(
        3.9, 0.07, r"$x_2=2.0\;\mathrm{fm}$", fontsize=10, color="r", ha="right"
    )
    ax[1].text(
        3.9, 0.04, r"$x_3=3.0\;\mathrm{fm}$", fontsize=10, color="r", ha="right"
    )
    ax[1].text(
        3.9, 0.01, r"$x_4=4.0\;\mathrm{fm}$", fontsize=10, color="r", ha="right"
    )

    # Plot trajectories for mu/T = 15
    initial_conditions = np.array([cfg.T_hat_0, 15 * cfg.T_hat_0, 0, 0, 0])
    solutions = solve_equations(initial_conditions, eom_instance)
    # plot_trajectories(args, cfg, fig, ax[2], solutions, eom_instance, initial_conditions, e_freezeout)
    ideal_solutions = solve_equations(
        initial_conditions, eom_instance, ideal_evolution=True
    )
    plot_ideal_trajectories(
        args,
        cfg,
        ax[2],
        fig,
        ideal_solutions,
        eom_instance,
        initial_conditions,
        e_freezeout,
    )

    xs = np.linspace(0, 40, 1000)
    mus = milne_mu(1.0, xs, 1.0, solutions[1][0])
    Ts = milne_T(1.0, xs, 1.0, solutions[0])
    ax[2].text(
        mus[0] + 0.05, Ts[0] - 0.02, r"$\tau_0$", fontsize=10, color="blue", ha="left"
    )
    ax[2].plot(mus, Ts, lw=1, color="blue", ls="dotted", zorder=8)
    mus = milne_mu(1.2, xs, 1.0, solutions[1][0])
    Ts = milne_T(1.2, xs, 1.0, solutions[0])
    ax[2].text(
        mus[0] + 0.05, Ts[0] - 0.02, r"$\tau_1$", fontsize=10, color="blue", ha="left"
    )
    ax[2].plot(mus, Ts, lw=1, color="blue", ls="dotted", zorder=8)
    mus = milne_mu(2.0, xs, 1.0, solutions[1][0])
    Ts = milne_T(2.0, xs, 1.0, solutions[0])
    ax[2].text(
        mus[0] + 0.1, Ts[0] - 0.02, r"$\tau_2$", fontsize=10, color="blue", ha="left"
    )
    ax[2].plot(
        mus, Ts, lw=1, color="blue", ls="dotted", zorder=8, label="Fixed proper time"
    )

    taus = np.linspace(1.0, 15, 1000)
    mus = milne_mu(taus, 0.0, 1.0, solutions[1][0])
    Ts = milne_T(taus, 0.0, 1.0, solutions[0])
    ax[2].text(mus[0], Ts[0] + 0.02, r"$x_0$", fontsize=10, color="r", ha="center")
    ax[2].scatter(mus[0], Ts[0], color="r", edgecolors="b", lw=0.5, s=10, zorder=10)
    ax[2].plot(mus, Ts, lw=1, color="red", ls="solid", zorder=9)
    mus = milne_mu(taus, 1.5, 1.0, solutions[1][0])
    Ts = milne_T(taus, 1.5, 1.0, solutions[0])
    ax[2].text(mus[0], Ts[0] + 0.02, r"$x_1$", fontsize=10, color="r", ha="center")
    ax[2].scatter(mus[0], Ts[0], color="r", edgecolors="b", lw=0.5, s=10, zorder=10)
    ax[2].plot(mus, Ts, lw=1, color="red", ls="solid", zorder=9)
    mus = milne_mu(taus, 2.0, 1.0, solutions[1][0])
    Ts = milne_T(taus, 2.0, 1.0, solutions[0])
    ax[2].text(mus[0], Ts[0] + 0.015, r"$x_2$", fontsize=10, color="r", ha="center")
    ax[2].scatter(mus[0], Ts[0], color="r", edgecolors="b", lw=0.5, s=10, zorder=10)
    ax[2].plot(mus, Ts, lw=1, color="red", ls="solid", zorder=9)
    mus = milne_mu(taus, 3.0, 1.0, solutions[1][0])
    Ts = milne_T(taus, 3.0, 1.0, solutions[0])
    ax[2].text(
        mus[0] - 0.05, Ts[0] + 0.02, r"$x_3$", fontsize=10, color="r", ha="center"
    )
    ax[2].scatter(mus[0], Ts[0], color="r", edgecolors="b", lw=0.5, s=10, zorder=10)
    ax[2].plot(mus, Ts, lw=1, color="red", ls="solid", zorder=9)
    mus = milne_mu(taus, 4.0, 1.0, solutions[1][0])
    Ts = milne_T(taus, 4.0, 1.0, solutions[0])
    ax[2].text(
        mus[0] - 0.05, Ts[0] + 0.02, r"$x_4$", fontsize=10, color="r", ha="center"
    )
    ax[2].scatter(mus[0], Ts[0], color="r", edgecolors="b", lw=0.5, s=10, zorder=10)
    ax[2].plot(mus, Ts, lw=1, color="red", ls="solid", zorder=9, label="Fixed position")

    ax[1].text(
        0.05, 0.42, r"$\tau_0=1.0\;\mathrm{fm}/c$", fontsize=10, color="b", ha="left"
    )
    ax[1].text(
        0.05, 0.39, r"$\tau_1=1.2\;\mathrm{fm}/c$", fontsize=10, color="b", ha="left"
    )
    ax[1].text(
        0.05, 0.36, r"$\tau_2=2.0\;\mathrm{fm}/c$", fontsize=10, color="b", ha="left"
    )

    ax[2].legend(loc="lower right", fontsize=10, frameon=False, handlelength=1.5)

    customize_axis(
        ax=ax[0],
        x_title=r"$\displaystyle r$ [fm]",
        y_title=r"$\displaystyle\tau_\mathrm{FO}$ [fm/$c$]",
        xlim=(-0.2, 6.2),
        ylim=(-0.2, 4.2),
    )
    ax[0].text(2.60, 0.55, r"$\displaystyle \hat{\mu}_{Y,0}/\hat{T}_0=0$", fontsize=10)
    ax[0].text(2.95, 2.1, r"$\displaystyle \hat{\mu}_{Y,0}/\hat{T}_0=10$", fontsize=10)
    ax[0].text(3.60, 3.25, r"$\displaystyle \hat{\mu}_{Y,0}/\hat{T}_0=15$", fontsize=10)

    customize_axis(
        ax=ax[1],
        x_title=r"$\displaystyle \mu_Y$ [GeV]",
        y_title=r"$\displaystyle T$ [GeV]",
        xlim=(-0.2, 4.2),
        ylim=(-0.02, 0.52),
    )

    ax[1].text(2.7, 0.25, "$\displaystyle s/n_Y=\\; $const.", rotation=38, fontsize=10)
    ax[1].text(
        0.95,
        0.91,
        r"$\displaystyle \hat{\mu}_{Y,0}/\hat{T}_0=10$",
        fontsize=10,
        ha='right',
        transform=ax[1].transAxes,
    )

    customize_axis(
        ax=ax[2],
        x_title=r"$\displaystyle \mu_Y$ [GeV]",
        y_title=r"$\displaystyle T$ [GeV]",
        xlim=(-0.2, 4.2),
        ylim=(-0.02, 0.52),
    )

    ax[2].text(
        0.95,
        0.91,
        r"$\displaystyle \hat{\mu}_{Y,0}/\hat{T}_0=15$",
        fontsize=10,
        ha='right',
        transform=ax[2].transAxes,
    )
    for name, label in zip(range(3), ["a", "b", "c"]):
        ax[name].text(
            0.215,
            0.91,
            r"\textsc{EoS1}",
            transform=ax[name].transAxes,
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "linewidth": 0.5},
            horizontalalignment="center",
        )
        ax[name].text(
            0.07,
            0.91,
            f"({label})",
            transform=ax[name].transAxes,
            fontsize=10,
            horizontalalignment="center",
        )

    # ax[1].set_aspect(6.428, anchor="SW")
    # ax[2].set_aspect(6.428, anchor="SW")
    fig.tight_layout()
    output_file = output_dir / "Fig3_VGCC-freezeout_and_trajectories.pdf"
    fig.savefig(output_file)
    logger.info(f"Saved figure to {output_file}")


if __name__ == "__main__":
    main()
