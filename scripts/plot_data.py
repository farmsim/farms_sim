"""Plot data"""

import os
import argparse

import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt

import farms_pylog as pylog
from farms_core.model.data import AnimatData
# from farms_core.model.options import AnimatOptions
from farms_core.simulation.options import SimulationOptions


plt.rc('axes', prop_cycle=(
    cycler(linestyle=['-', '--', '-.', ':'])
    * cycler(color=plt.rcParams['axes.prop_cycle'].by_key()['color'])
))


def parse_args():
    """Parse args"""
    parser = argparse.ArgumentParser(
        description='Plot amphibious simulation data',
        formatter_class=(
            lambda prog:
            argparse.HelpFormatter(prog, max_help_position=50)
        ),
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Data',
    )
    # parser.add_argument(
    #     '--animat',
    #     type=str,
    #     help='Animat options',
    # )
    parser.add_argument(
        '--simulation',
        type=str,
        help='Simulation options',
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path',
    )
    parser.add_argument(
        '--drive_config',
        type=str,
        default='',
        help='Descending drive method',
    )
    return parser.parse_args()


def main():
    """Main"""

    # Clargs
    clargs = parse_args()

    # Load data
    # animat_options = AnimatOptions.load(clargs.animat)
    simulation_options = SimulationOptions.load(clargs.simulation)
    animat_data = AnimatData.from_file(clargs.data)

    # Plot simulation data
    times = np.arange(
        start=0,
        stop=simulation_options.timestep*simulation_options.n_iterations,
        step=simulation_options.timestep,
    )
    assert len(times) == simulation_options.n_iterations
    plots_sim = animat_data.plot_sensors(times)

    # Save plots
    extension = 'pdf'
    for name, fig in {**plots_sim}.items():
        filename = os.path.join(clargs.output, f'{name}.{extension}')
        pylog.debug('Saving to %s', filename)
        fig.savefig(filename, format=extension, bbox_inches='tight')


if __name__ == '__main__':
    main()
