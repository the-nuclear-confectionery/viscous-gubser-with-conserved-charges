#!/usr/bin/env python3
"""
Plots the energy density from VGCC_initial_condition.dat as a function of x and y.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from scipy.interpolate import griddata
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_data(file_path: str) -> pd.DataFrame:
    """
    Read data from a file, ignoring lines starting with '#' or '//'.

    Args:
        file_path (str): Path to the data file.

    Returns:
        pd.DataFrame: DataFrame containing the data.
    """
    # Load data ignoring lines starting with '#' or '//'
    data = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None)

    # Extract relevant columns
    x = data[0]
    y = data[1]
    e = data[3]
    nB = data[4]
    ux = data[7]
    uy = data[8]

    return x, y, e, nB, ux, uy

def interpolate_data(x: np.ndarray, y: np.ndarray, e:np.ndarray, nB: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> tuple:
    """
    Interpolate data for plotting.

    Args:
        x (np.ndarray): x-coordinates.
        y (np.ndarray): y-coordinates.
        e (np.ndarray): Energy density.
        nB (np.ndarray): Number density density.
        ux (np.ndarray): x-component of velocity.
        uy (np.ndarray): y-component of velocity.

    Returns:
        tuple: Interpolated data for plotting.
    """
    # Create a grid for x and y
    xi = np.linspace(x.min(), x.max(), 1001)
    yi = np.linspace(y.min(), y.max(), 1001)
    xi, yi = np.meshgrid(xi, yi)
    # Interpolate the energy density over the grid.
    ei = griddata((x, y), e, (xi, yi), method='linear')
    # Interpolate the nucleon number density (nB) over the grid.
    nBi = griddata((x, y), nB, (xi, yi), method='linear')
    # Mask out all points below the threshold
    nB_masked = np.ma.array(nBi, mask=(ei < 1.0))
    # Interpolate the velocity components over the same grid.
    ux_grid = griddata((x, y), ux, (xi, yi), method='linear')
    uy_grid = griddata((x, y), uy, (xi, yi), method='linear')

    return xi, yi, nB_masked, ux_grid, uy_grid


def plot_energy_density(input_dir: Path, times: list) -> None:
    """
    Plot the energy density from the data file.
    """
    # Check if the input directory exists
    if not input_dir.is_dir():
        logger.error("input_dir must be a directory.")
        sys.exit(1)
    # Check if the list of times is empty
    if len(times) == 0:
        logger.error("times list cannot be empty.")
        sys.exit(1)
    # Check if the times are valid
    valid_times = []
    for time in times:
        try:
            data_file = Path(input_dir) / f"VGCC_tau={time}.dat"
            if not data_file.is_file():
                logger.warning(f"File {data_file} does not exist. Skipping time {time}.")
            else:
                valid_times.append(time)
        except Exception as e:
            logger.error(f"Invalid time format: {time}. Must be a number. Error: {e}")
            sys.exit(1)

    if not valid_times:
        logger.error("No valid times were found. Exiting.")
        sys.exit(1)

    # Prepare to collect interpolated data and global min/max
    interp_results = []  # List of tuples: (x, y, xi, yi, zi_masked, ux_grid, uy_grid)
    global_vmin = np.inf
    global_vmax = -np.inf

    # First pass: interpolate data & determine global energy density min/max.
    for time in valid_times:
        logger.info(f"Processing time: {time} fm/c")
        data_file = Path(input_dir) / f"VGCC_tau={time}.dat"
        logger.info(f"Reading data from {data_file}")
        x, y, e, nB, ux, uy = read_data(data_file)
        logger.info("Interpolating data")
        xi, yi, nB_masked, ux_grid, uy_grid = interpolate_data(x, y, e, nB, ux, uy)
        
        # Update the global minimum and maximum (use np.ma.min and np.ma.max to ignore masked values)
        local_vmin = np.ma.min(nB_masked)
        local_vmax = np.ma.max(nB_masked)
        global_vmin = min(global_vmin, local_vmin)
        global_vmax = max(global_vmax, local_vmax)
        
        interp_results.append((x, y, xi, yi, nB_masked, ux_grid, uy_grid))

    # Create a figure with one axis per valid time.
    fig, axs = plt.subplots(1, len(valid_times), figsize=(5*len(valid_times), 5), dpi=300)
    # Set the background color
    background_color = 'black'
    fig.set_facecolor(background_color)
    if len(valid_times) == 1:
        # Ensure axs is iterable if only one axis exists.
        axs = [axs]
    for ax in axs:
        ax.set_xlim(x.min()/2, x.max()/2)
        ax.set_ylim(y.min()/2, y.max()/2)
        ax.set_facecolor(background_color)
        ax.axis('off')

    # Second pass: plot the results using the global vmin and vmax.
    for i, time in enumerate(valid_times):
        x, y, xi, yi, nB_masked, ux_grid, uy_grid = interp_results[i]
        logger.info(f"Plotting time: {time} fm/c")
        axs[i].set_title(r"$\tau=$" + f' {time} ' + r"fm/$c$", fontsize=22, color='white')
        axs[i].imshow(nB_masked, extent=[x.min(), x.max(), y.min(), y.max()],
                      origin='lower', cmap='plasma',
                      norm=colors.PowerNorm(gamma=0.5, vmin=global_vmin, vmax=global_vmax))
        logger.info("Plotting velocity field")
        stride = 100    # Stride to reduce the number of quiver arrows.
        axs[i].quiver(xi[::stride, ::stride],
                      yi[::stride, ::stride],
                      ux_grid[::stride, ::stride],
                      uy_grid[::stride, ::stride],
                      color='white',
                      width=0.01,
                      alpha=0.6,
                      scale=10,
                      pivot='mid')

    # Save the figure
    output_file = "VGCC_evolution.png"
    fig.savefig(output_file, bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.close()
    logger.info(f"Figure saved to {output_file}")

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(prog="VISCOUS GUBSER FLOW WITH CONSERVED CHARGES (VGCC)", 
                                     description="""Tool to visualize the evolution of number density
                                                    and velocity field as a function of time.""",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog="Developed by the UIUC Nuclear Theory Group.")
    parser.add_argument("--input-dir", "-i", 
                        required=True,
                        type=Path,
                        help="path to Gubser event data files directory.")
    parser.add_argument("--times", "-t",
                        required=True,
                        nargs='+',
                        help="list of times to plot.")

    return parser.parse_args()

def main() -> None:
    """
    Main entry point for the script.
    """
    # Parse command-line arguments.
    args = parse_args()

    input_dir = args.input_dir
    times = args.times
    plot_energy_density(input_dir, times)

if __name__ == "__main__":
    main()