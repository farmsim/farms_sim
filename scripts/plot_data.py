"""Plot data"""

import os
import argparse

import numpy as np

import farms_pylog as pylog
from farms_data.amphibious.animat_data import AnimatData
from farms_bullet.simulation.options import SimulationOptions
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.utils.network import plot_networks_maps
from farms_amphibious.control.drive import drive_from_config, plot_trajectory


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
    animat_options = AmphibiousOptions.load(clargs.animat)
    simulation_options = SimulationOptions.load(clargs.simulation)
    animat_data = AnimatData.from_file(clargs.data)

    # Plot simulation data
    times = np.arange(
        start=0,
        stop=simulation_options.timestep*simulation_options.n_iterations,
        step=simulation_options.timestep,
    )
    assert len(times) == simulation_options.n_iterations
    plots_sim = animat_data.plot(times)

    # Plot connectivity
    plots_network = (
        plot_networks_maps(
            morphology=animat_options.morphology,
            data=animat_data,
            show_all=True,
        )[1]
        if animat_options.morphology.n_dof_legs == 4
        else {}
    )

    # Plot descending drive
    if clargs.drive_config:
        pos = np.array(animat_data.sensors.links.urdf_positions()[:, 0])
        drive = drive_from_config(
            filename=clargs.drive_config,
            animat_data=animat_data,
            simulation_options=simulation_options,
        )
        fig3 = plot_trajectory(drive.strategy, pos)
        plots_drive = {'trajectory': fig3}
    else:
        plots_drive = {}

    # Save plots
    extension = 'pdf'
    for name, fig in {**plots_sim, **plots_network, **plots_drive}.items():
        filename = os.path.join(clargs.output, '{}.{}'.format(name, extension))
        fig.savefig(filename, format=extension)
        pylog.debug('Saving to %s', filename)


if __name__ == '__main__':
    main()
