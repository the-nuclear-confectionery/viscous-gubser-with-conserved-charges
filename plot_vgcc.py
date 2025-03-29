#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        "--analytical_path",
        default="data/output/analytic_solutions/EoS2",
        help="folder containing the analytical solutions.",
    )
    parser.add_argument(
        "--simulation_path",
        default="data/CCAKE_results/",
        help="folder containing the CCAKE results.",
    )
    return parser.parse_args()


def main():
    """
    Main entry point for the script.
    """
    # Parse command-line arguments.
    args = parse_args()

    # Define the list of plotting scripts and their arguments.
    # Adjust the script names and arguments as needed.
    scripts_to_run = [
        {
            "script": os.path.join("plots", "plot_Fig1_VGCC_T_mu_pi.py"),
            "args": ["--output_path", "data/output/plots/", "--debug"],
        },
        {
            "script": os.path.join(
                "plots", "plot_Fig3_VGCC_freezeout_and_trajectories.py"
            ),
            "args": ["--output_path", "data/output/plots/", "--debug"],
        },
        {
            "script": os.path.join("plots", "plot_Fig4_CCAKE_vs_VGCC_e_n_u_pi_Rn.py"),
            "args": [
                "--analytical_path",
                args.analytical_path,
                "--simulation_path",
                args.simulation_path,
                "--output_path",
                "data/output/plots/",
            ],
        },
        {
            "script": os.path.join("plots", "plot_Fig_CCAKE_2.0_Gubser.py"),
            "args": [
                "--analytical_path",
                args.analytical_path,
                "--simulation_path",
                args.simulation_path,
                "--output_path",
                "data/output/plots/",
            ],
        },
        {
            "script": os.path.join("plots", "plot_Fig5_CCAKE_vs_VGCC_hypersurface.py"),
            "args": [
                "--simulation_path",
                args.simulation_path,
                "--output_path",
                "data/output/plots/",
                "--debug",
            ],
        },
    ]

    for entry in scripts_to_run:
        script_path = entry["script"]
        args = entry["args"]

        if not os.path.exists(script_path):
            logger.info(f"Script not found: {script_path}")
            continue

        cmd = ["python3", script_path] + args
        logger.info(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Finished running {script_path}\n")
        except subprocess.CalledProcessError as err:
            logger.error(f"Error running {script_path}: {err}")
            sys.exit(1)


if __name__ == "__main__":
    main()
