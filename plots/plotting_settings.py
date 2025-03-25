#!/usr/bin/env python3
"""
Authors: Kevin Ingles, Jordi Salinas
File: plotting_settings.py
Description: User-defined functions to facilitate plotting routines.

This module provides helper functions for customizing matplotlib plots, including:
    - Retrieving a colormap with a given number of discrete colors.
    - Customizing axis labels and tick parameters.
    - Autoscaling the y-axis based on the data visible in the current x-axis range.
    - Smoothing a set of data points via cubic spline interpolation.
"""

from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib import rc
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from scipy.interpolate import CubicSpline  # Used in smooth_histogram

# Update matplotlib rcParams for LaTeX usage and preferred fonts.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern Roman",
    "text.latex.preamble": r'\usepackage{amsmath}',
    # "axes.labelsize": 14,
    # "font.size": 14,
    # "xtick.labelsize": 14,
    # "ytick.labelsize": 10,
})


def get_cmap(name: str, n: int) -> Colormap:
    """
    Return a colormap instance that maps each index in 0, 1, ..., n-1 to a distinct RGB color.

    Parameters:
        n (int): Number of discrete colors needed.
        name (str, optional): The name of the matplotlib colormap to use. Defaults to 'hsv'.

    Returns:
        Colormap: A matplotlib colormap instance with n discrete colors.
    """
    return plt.cm.get_cmap(name, n)


def customize_axis(ax: Axes, 
                   x_title: str, 
                   y_title: str, 
                   no_xnums: bool = False,
                   xlim: Optional[Tuple[float, float]] = None,
                   ylim: Optional[Tuple[float, float]] = None) -> Axes:
    """
    Customize a matplotlib Axes object by setting axis labels, tick parameters, 
    and optionally the axis limits.

    The function sets the x- and y-axis labels with a font size of 14 and configures 
    both major and minor tick parameters. Optionally, it can hide the x-axis tick labels.
    Additionally, if x-axis or y-axis limits are provided, they will be applied to the axis.

    Parameters:
        ax (Axes): The matplotlib Axes object to customize.
        x_title (str): The label for the x-axis.
        y_title (str): The label for the y-axis.
        no_xnums (bool, optional): If True, hides the x-axis numerical tick labels. Defaults to False.
        xlim (Optional[Tuple[float, float]], optional): A tuple (lower, upper) specifying the x-axis limits.
                                                         If provided, sets the x-axis limits. Defaults to None.
        ylim (Optional[Tuple[float, float]], optional): A tuple (lower, upper) specifying the y-axis limits.
                                                         If provided, sets the y-axis limits. Defaults to None.

    Returns:
        Axes: The modified Axes object.
    """
    ax.set_xlabel(x_title, fontsize=14)
    ax.set_ylabel(y_title, fontsize=14)
    
    if no_xnums:
        # Hide x-axis tick labels while keeping tick marks for a neat appearance.
        ax.set_xticklabels([])  # clear tick labels on x-axis
        ax.tick_params(axis='x', top=True)
        ax.tick_params(axis='y', labelsize=14, right=True)
    else:
        ax.tick_params(axis='both', labelsize=14, top=True, right=True)
    
    # Configure major ticks: inward direction and a longer tick length.
    ax.tick_params(axis='both', which='major', direction='in', length=4)
    
    # Set up minor tick locators and configure their appearance.
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.tick_params(axis='both', which='minor', direction='in', length=2, top=True, right=True)
    
    # If provided, set the axis limits.
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    return ax


def autoscale_y(ax: Axes, margin: float = 0.1) -> None:
    """
    Rescale the y-axis limits based on the data currently visible within the x-axis limits.

    For each line on the given axis, the function considers only the data points that lie within
    the current x-axis limits. It computes the minimum and maximum y-values for these points and
    then adjusts the y-axis limits by adding a specified fractional margin to the range.

    Parameters:
        ax (Axes): A matplotlib Axes object whose y-axis will be rescaled.
        margin (float, optional): Fraction of the data range to pad on both the top and bottom.
                                  Defaults to 0.1.

    Returns:
        None
    """
    def get_bottom_top(line: plt.Line2D) -> Optional[Tuple[float, float]]:
        """
        Compute the bottom and top y-values for a given line based on the currently visible x-range.

        Parameters:
            line (Line2D): A matplotlib Line2D object from the Axes.

        Returns:
            Optional[Tuple[float, float]]: A tuple (bottom, top) with computed limits,
            or None if no data points of the line are within the current x-axis limits.
        """
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo, hi = ax.get_xlim()
        # Select only the data points within the current x-axis limits.
        mask = (xd > lo) & (xd < hi)
        y_displayed = yd[mask]
        if y_displayed.size == 0:
            return None
        y_min = np.min(y_displayed)
        y_max = np.max(y_displayed)
        h = y_max - y_min
        return y_min - margin * h, y_max + margin * h

    lines = ax.get_lines()
    if not lines:
        print('No lines in plot, leaving plot unchanged.')
        return

    overall_bot, overall_top = np.inf, -np.inf

    # Iterate over each line to determine the overall bottom and top limits.
    for line in lines:
        limits = get_bottom_top(line)
        if limits is None:
            continue
        new_bot, new_top = limits
        overall_bot = min(overall_bot, new_bot)
        overall_top = max(overall_top, new_top)

    if np.isinf(overall_bot) or np.isinf(overall_top):
        print("Infinite limits encountered, leaving y-axis unchanged: check for error.")
        return

    ax.set_ylim(overall_bot, overall_top)


def smooth_histogram(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth a set of data points using cubic spline interpolation.

    This function generates a new, denser array of x-values (with twice the number of original points)
    and computes corresponding y-values via a cubic spline interpolation, which can be useful for smoothing
    histograms or other noisy data.

    Parameters:
        x (np.ndarray): 1D array of x-values.
        y (np.ndarray): 1D array of y-values corresponding to the x-values.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - xs (np.ndarray): The new, denser array of x-values.
            - ys (np.ndarray): The interpolated y-values corresponding to xs.
    """
    low, high = x[0], x[-1]
    xs = np.linspace(low, high, x.size * 2)
    cs = CubicSpline(x, y)
    return xs, cs(xs)


if __name__ == "__main__":
    # === Test get_cmap ===
    cmap = get_cmap(5, name='viridis')
    print("Test get_cmap: Retrieved colormap:", cmap.name)

    # === Test customize_axis ===
    fig, ax = plt.subplots(figsize=(6, 4))
    x_vals = np.linspace(0, 10, 100)
    ax.plot(x_vals, np.sin(x_vals), label="sin(x)")
    customize_axis(ax, "X Axis", "Y Axis", no_xnums=False)
    ax.legend()
    ax.set_title("Test of customize_axis")
    plt.tight_layout()
    plt.show()

    # === Test autoscale_y ===
    fig, ax = plt.subplots(figsize=(6, 4))
    x_vals = np.linspace(0, 10, 100)
    ax.plot(x_vals, np.sin(x_vals), label="sin(x)")
    # Set a limited x-range to trigger autoscaling based on visible data.
    ax.set_xlim(2, 8)
    autoscale_y(ax, margin=0.1)
    ax.set_title("Test of autoscale_y")
    plt.tight_layout()
    plt.show()

    # === Test smooth_histogram ===
    x_orig = np.linspace(0, 10, 20)
    y_orig = np.random.random(20)
    xs_smooth, ys_smooth = smooth_histogram(x_orig, y_orig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_orig, y_orig, 'o', label="Original data")
    ax.plot(xs_smooth, ys_smooth, '-', label="Smoothed data")
    customize_axis(ax, "X Axis", "Y Axis")
    ax.legend()
    ax.set_title("Test of smooth_histogram")
    plt.tight_layout()
    plt.show()
