"""Prompt"""

import os
import argparse
from distutils.util import strtobool

import matplotlib.pyplot as plt

import farms_pylog as pylog
from farms_models.utils import get_simulation_data_path
from farms_data.amphibious.animat_data import AnimatData
from farms_amphibious.utils.network import plot_networks_maps


def prompt(query, default):
    """Prompt"""
    val = input('{} [{}]: '.format(
        query,
        'Y/n' if default else 'y/N',
    ))
    try:
        ret = strtobool(val) if val != '' else default
    except ValueError:
        print('Dit not recognise \'{}\', please reply with a y/n'.format(val))
        return prompt(query, default)
    return ret


def parse_args():
    """Parse args"""
    parser = argparse.ArgumentParser(
        description='FARMS amphibious',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-p', '--prompt',
        dest='prompt',
        action='store_true',
        help='Prompt at end of simulation',
    )
    return parser.parse_known_args()


def prompt_postprocessing(
        animat,
        version,
        sim,
        animat_options,
):
    """Prompt postprocessing"""
    # Post-processing
    pylog.info('Simulation post-processing')
    log_path = get_simulation_data_path(
        name=animat,
        version=version,
        simulation_name='default',
    )
    video_name = os.path.join(log_path, 'simulation.mp4')
    save_data = prompt('Save data', False)
    if log_path and not os.path.isdir(log_path):
        os.mkdir(log_path)
    show_plots = prompt('Show plots', False)
    sim.postprocess(
        iteration=sim.iteration,
        log_path=log_path if save_data else '',
        plot=show_plots,
        video=video_name if sim.options.record else ''
    )
    if save_data:
        pylog.debug('Data saved, now loading back to check validity')
        data = AnimatData.from_file(os.path.join(log_path, 'simulation.hdf5'))
        pylog.debug('Data successfully saved and logged back: {}'.format(data))

    # Plot network
    show_connectivity = prompt('Show connectivity maps', False)
    if show_connectivity:
        plot_networks_maps(animat_options.morphology, sim.animat().data)

    # Plot
    if (show_plots or show_connectivity) and prompt('Save plots', False):
        extension = 'pdf'
        for fig in [plt.figure(num) for num in plt.get_fignums()]:
            filename = '{}.{}'.format(
                os.path.join(log_path, fig.canvas.get_window_title()),
                extension,
            )
            filename = filename.replace(' ', '_')
            pylog.debug('Saving to {}'.format(filename))
            fig.savefig(filename, format=extension)
    if show_plots or (
            show_connectivity
            and prompt('Show connectivity plots', False)
    ):
        plt.show()
