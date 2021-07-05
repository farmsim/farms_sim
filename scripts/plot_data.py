"""Plot data"""

import os
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from PIL import Image
from moviepy.editor import VideoClip

import farms_pylog as pylog
from farms_data.amphibious.animat_data import AnimatData
from farms_bullet.simulation.options import SimulationOptions
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.utils.network import plot_networks_maps


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
    parser.add_argument(
        '--animat',
        type=str,
        help='Animat options',
    )
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
    return parser.parse_args()


def main():
    """Main"""

    # Clargs
    clargs = parse_args()

    # Load data
    animat_options = AmphibiousOptions.load(clargs.animat)
    simulation_options = SimulationOptions.load(clargs.simulation)
    sim_data = AnimatData.from_file(clargs.data)

    # Plot simulation data
    times = np.arange(
        start=0,
        stop=simulation_options.timestep*simulation_options.n_iterations,
        step=simulation_options.timestep,
    )
    assert len(times) == simulation_options.n_iterations
    plots = sim_data.plot(times)

    # Plot connectivity
    morph = animat_options.morphology
    if morph.n_legs == 4 and morph.n_dof_legs == 4:
        plot_networks_maps(animat_options.morphology, sim_data, show_all=True)

    # Save plots
    extension = 'jpg'
    for name, fig in plots.items():
        filename = os.path.join(clargs.output, '{}.{}'.format(name, extension))
        pylog.debug('Saving to {}'.format(filename))
        fig.savefig(filename, format=extension)


if __name__ == '__main__':
    main()
