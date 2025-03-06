import sys
import os
import argparse

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox, TransformedBbox
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.constants import HBARC


import plotting_settings as myplt  # noqa


plot_index = 1
analytical_style = {'ls': '-', 'lw': 2}
# sim_style = {'ls':'-.','lw':3.,'alpha':.5}
# sim_style = {'facecolors': 'none'}

time_list = np.arange(1.00, 1.70, 0.1)  # Use this to focus on before FO
# time_list=['1.00', '1.20',  '1.40',  '1.60']# Use this to focus on before FO
# time_list=['1.00', '1.50', '2.00', '3.00', '4.00', '5.00', '6.00', '7.00', '8.00', '9.00', '10.00' ]
filter_criteria = 'abs(phi - 3.141592653589793/4.) < 1.e-2'

cmap = myplt.get_cmap('cividis', len(time_list))

mpl.rcParams['text.usetex'] = True

dpi = 1200
fig, ax = plt.subplot_mosaic([['e', 'rhoB', 'ux', 'cbar'],
                              ['pixx', 'Rey', 'pixy', 'cbar']],
                             width_ratios=[1, 1, 1, 0.35],
                             figsize=np.array([10.5 , 6]),
                            #  sharex=True,
                             dpi=dpi,
                             constrained_layout=True)


def get_reynolds_number(df, t_squared):
    df['u0'] = np.sqrt(1 + df['ux']**2 + df['uy']**2)
    df['pitx'] = (df['pixx'] * df['ux'] + df['pixy'] * df['uy']) / df['u0']
    df['pity'] = (df['pixy'] * df['ux'] + df['piyy'] * df['uy']) / df['u0']
    df['pitt'] = (df['pitx'] * df['ux'] + df['pity'] * df['uy']) / df['u0']
    df['pizz'] = -(df['pixx'] + df['piyy'] - df['pitt']) / t_squared
    df['p'] = df['e'] / 3  # Pressure

    df['pi_norm'] = np.sqrt(df['pitt']**2 + df['pixx']**2 + df['piyy']**2
                            + (t_squared * df['pizz'])**2
                            + 2 * (df['pixy']**2 - df['pitx']**2
                                   - df['pity']**2))
    df['reynolds'] = df['pi_norm'] / df['p']
    return df


def read_sol(analytical_sol_folder):
    for ii, t in enumerate(time_list):
        if ii % plot_index != 0:
            continue
        inp_path = os.path.join(analytical_sol_folder, f'VGCC_tau={t:.2f}' '.dat')
        df = pd.read_table(
            inp_path,
            names=[
                'x',
                'y',
                'eta',
                'e',
                'rhoB',
                'rhoS',
                'rhoQ',
                'ux',
                'uy',
                'ueta',
                'Bulk',
                'pixx',
                'pixy',
                'pixeta',
                'piyy',
                'pyeta',
                'pietaeta'],
            sep="\s+",
            engine='python',
            header=1)

        df['r'] = np.sqrt(df['x']**2 + df['y']**2)
        df['phi'] = np.arctan2(df['y'], df['x'])
        df = get_reynolds_number(df, float(t)**2)

        df = df.query(filter_criteria)

        ax['e'].plot(df['r'], df['e'], color=cmap(ii), **analytical_style)
        ax['rhoB'].plot(df['r'], df['rhoB'], color=cmap(ii), **analytical_style)
        ax['ux'].plot(df['r'], df['ux'], color=cmap(ii), **analytical_style)
        ax['pixx'].plot(df['r'], df['pixx'], color=cmap(ii), **analytical_style)
        ax['Rey'].plot(
            df['r'],
            df['reynolds'],
            color=cmap(ii),
            **analytical_style)
        ax['pixy'].plot(
            df['r'],
            df['pixy'],
            color=cmap(ii),
            **analytical_style)


def read_sim(sim_result_folder):
    dt = .001
    for ii, t in enumerate(time_list):
        if ii % plot_index != 0:
            continue
        col_names = [
            'id',
            't',
            'x',
            'y',
            'p',
            'T',
            'muB',
            'muS',
            'muQ',
            'e',
            'rhoB',
            'rhoS',
            'rhoQ',
            's',
            's_smoothed',
            's_specific',
            'sigma',
            'spec_s',
            'stauRelax',
            'bigTheta',
            '??',
            '??2',
            'pi00',
            'pixx',
            'piyy',
            'pixy',
            't2pi33',
            'v1',
            'v2',
            'gamma',
            'frz',
            'eos']
        idx = int(np.round((float(t) - 1) / dt) / 100)
        inp_path = os.path.join(sim_result_folder, f'system_state_{idx}.dat')
        print(inp_path)
        df = pd.read_table(inp_path,
                           names=col_names, sep=' ', header=0)
        df['ux'] = df.loc[:, 'v1'] * df.loc[:, 'gamma']
        df['uy'] = df.loc[:, 'v2'] * df.loc[:, 'gamma']
        df['r'] = np.sqrt(df['x']**2 + df['y']**2)
        df['phi'] = np.arctan2(df['y'], df['x'])
        df['e'] = df['e'] / 1000  # convert to GeV/fm^3

        df['pixx'] = df['pixx'] * HBARC  # convert to GeV/fm^3
        df['pixy'] = df['pixy'] * HBARC  # convert to GeV/fm^3
        df['piyy'] = df['piyy'] * HBARC  # convert to GeV/fm^3

        df['t2pi33'] = df['t2pi33'] * HBARC  # convert to GeV/fm^3

        df = get_reynolds_number(df, float(t)**2)
        df_query = df.query(filter_criteria)

        # df_query = df_query[::3]

        sim_style = {'facecolor': cmap(ii), 's': 20, 'edgecolors': 'k'}

        stride = 3
        ax['e'].scatter(
            df_query['r'].to_numpy()[
                ::stride], df_query['e'].to_numpy()[
                ::stride], **sim_style, zorder=2, lw=0.5)
        ax['rhoB'].scatter(
            df_query['r'].to_numpy()[
                ::stride], df_query['rhoB'].to_numpy()[
                ::stride], **sim_style, zorder=2, lw=0.5)
        ax['ux'].scatter(
            df_query['r'].to_numpy()[
                ::stride], df_query['ux'].to_numpy()[
                ::stride], **sim_style, zorder=2, lw=0.5)
        ax['pixx'].scatter(
            df_query['r'].to_numpy()[
                ::stride], df_query['pixx'].to_numpy()[
                ::stride], **sim_style, zorder=2, lw=0.5)
        ax['Rey'].scatter(
            df_query['r'].to_numpy()[
                ::stride], df_query['reynolds'].to_numpy()[
                ::stride], **sim_style, zorder=2, lw=0.5)
        ax['pixy'].scatter(
            df_query['r'].to_numpy()[
                ::stride], df_query['pixy'].to_numpy()[
                ::stride],  **sim_style, zorder=2, lw=0.5)


def beautify():
    # fig.set_tight_layout(True)
    ylabels = {'e': r'$\displaystyle \mathcal E$ [GeV/fm$^3$]',
               'rhoB': r'$\displaystyle n_Y$ [fm$^{-3}$]',
               'ux': r'$\displaystyle u^x$',
               'pixx': r'$\displaystyle  \pi^{xx}$ [GeV/fm$^3$]',
               'Rey': r'$\displaystyle  \mathcal{R}^{-1}$',
               # 'pixy':r'$\pi^{xy}$ (GeV/fm$^3$)',
               'pixy': r'$\displaystyle \pi^{xy}$ [GeV/fm$^3$]'}
    for key in ax.keys():
        if key == 'cbar':
            continue
        elif key in ['pixx', 'Rey', 'pixy']:
            if key == 'pixx':
                ylim = (-0.45,0.05)
            elif key == 'Rey':
                ylim = (-0.2, 3.2)
            elif key == 'pixy':
                ylim = (-0.16, 0.01)
            myplt.customize_axis(ax=ax[key],
                                 x_title=r'$\displaystyle r$ [fm]',
                                 y_title=ylabels[key],
                                 xlim=(-0.2, 4.2),
                                 ylim=ylim)
        else:
            if key == 'e':
                ylim = (-0.25, 8.25)
            elif key == 'rhoB':
                ylim = (-0.25, 8.25)
            elif key == 'ux':
                ylim = (-0.1, 1.6)
            myplt.customize_axis(ax=ax[key],
                                    x_title=r'$\displaystyle r$ [fm]',
                                    y_title=ylabels[key],
                                    xlim=(-0.2, 4.2),
                                    ylim=ylim)

    # myplt.customize_axis(ax=ax['cbar'], x_title='', y_title='')

    tau_min = float(time_list[0])
    tau_max = float(time_list[-1])
    tau_list = [float(t) for t in time_list]

    # deal with the colorbar
    delta_time = tau_list[1] - tau_list[0]
    boundaries = np.arange(tau_min - delta_time / 2,
                           tau_max + 3 * delta_time / 2, delta_time)

    norm = mpl.colors.Normalize(vmin=tau_min, vmax=tau_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add the colorbar to the figure
    ax['cbar'].axis('off')
    x0, y0, width, height = [1.275, -0.032, 0.17, 1.1787] # if placed horizontally
    # and transform them after to get the ABSOLUTE POSITION AND DIMENSIONS
    box = Bbox.from_bounds(x0, y0, width, height)
    trans = ax['cbar'].transAxes + fig.transFigure.inverted() # pyright: ignore
    l, b, w, h = TransformedBbox(box, trans).bounds
    # Now just create the axes and the colorbar
    cbaxes = fig.add_axes([l, b, w, h])
    cbar = fig.colorbar(sm, ticks=tau_list, ax=ax['cbar'], cax=cbaxes, orientation='vertical', boundaries=boundaries)
    cbar.ax.tick_params(direction='out', labelsize=14)
    cbaxes.xaxis.set_label_position('top')
    cbaxes.xaxis.set_ticks_position('top')
    cbar.set_label(label=r'$\displaystyle \tau$ [fm/$c$]', size=14, labelpad=10)

    


    # cbar = plt.colorbar(sm, ticks=tau_list, label=r'$\displaystyle \tau$ [fm/$c$]',
    #                     cax=ax['cbar'],
    #                     boundaries=boundaries)
    # # Set ticks to point outward.
    # cbar.ax.tick_params(direction="out")

    ax['e'].plot([], [], **analytical_style, label='Semi-analytical', color=cmap(0))
    ax['e'].scatter([], [], label=r'\textsc{CCAKE}', edgecolors=cmap(0))
    ax['e'].text(0.75, 0.90, r'\textsc{EoS2}', transform=ax['e'].transAxes,
        fontsize=10, bbox={'facecolor': 'white', 'boxstyle': 'round', 'linewidth': 0.5},
        horizontalalignment='center',
    )


    # ax['ux'].set_ylim(0,3.3)
    # ax['ux'].set_ylim(0, 1.3)
    # ax['e'].set_ylim(0, 8)
    # ax['Rey'].set_ylim(0, 3.1)

    for name, label in zip(ylabels.keys(),
                           ['a', 'b', 'c', 'd', 'e', 'f']):
        if label in ['a', 'b', 'c']:
            xpos = 0.92
            ypos = 0.90
        else:
            xpos = 0.92
            ypos = 0.06
        ax[name].text(
            xpos,
            ypos,
            f'({label})',
            transform=ax[name].transAxes,
            fontsize=12,
            # bbox={'boxstyle': 'round', 'facecolor': 'white'},
            horizontalalignment='center',
        )

    leg = ax['e'].legend(loc='upper left', fontsize=10, bbox_to_anchor=(0.37,0.88), markerscale=12, frameon=False)
    LH = leg.legend_handles
    LH[0].set_linewidth(1.5)
    LH[0].set_color('k')
    LH[1].set_facecolor('white')
    LH[1].set_edgecolor('k')
    LH[1].set_linewidth(0.5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Plot Fig4 CCAKE vs VGCC: e, n, u, π, Reynolds number."
    )
    parser.add_argument(
        "--analytical_path",
        required=True,
        help="Path to the folder containing the analytical solutions."
    )
    parser.add_argument(
        "--simulation_path",
        required=True,
        help="Path to the folder containing the CCAKE results."
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path (excluding filename) to save the figure."
    )
    args = parser.parse_args()

    # Define the subdirectories for analytical and simulation data.
    filename = "Fig4_CCAKE-vs-VGCC.pdf"
    output_path = os.path.join(args.output_path, filename)

    read_sol(args.analytical_path)
    read_sim(args.simulation_path)
    beautify()

    fig.savefig(output_path)
