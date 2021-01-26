"""Prompt"""

import os
from distutils.util import strtobool

import matplotlib.pyplot as plt

import farms_pylog as pylog
from farms_models.utils import get_simulation_data_path
from farms_data.amphibious.animat_data import AnimatData
from farms_amphibious.utils.network import plot_networks_maps
from farms_bullet.simulation.parse_args import argument_parser


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
    parser = argument_parser()
    parser.description = 'Salamander simulation'
    parser.add_argument(
        '-p', '--prompt',
        action='store_true',
        help='Prompt at end of simulation',
    )
    parser.add_argument(
        '-s', '--save',
        type=str,
        default='',
        help='Save simulation data to provided path',
    )
    parser.add_argument(
        '--models',
        action='store_true',
        help='Save data to farms_models_data',
    )
    parser.add_argument(
        '--profile',
        type=str,
        default='simulation.profile',
        help='Save simulation profile to given filename',
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test simulation configuratioin files',
    )
    return parser.parse_args()


def prompt_postprocessing(
        animat,
        version,
        sim,
        animat_options,
        query=True,
        **kwargs
):
    """Prompt postprocessing"""
    # Arguments
    save = kwargs.pop('save', '')
    models = kwargs.pop('models', '')
    extension = kwargs.pop('extension', 'pdf')

    # Post-processing
    pylog.info('Simulation post-processing')
    log_path = get_simulation_data_path(
        name=animat,
        version=version,
        simulation_name=save if save else 'default',
    ) if models else save
    save_data = (
        (query and prompt('Save data', False))
        or (save or models) and not query
    )
    if log_path and not os.path.isdir(log_path):
        os.mkdir(log_path)
    show_plots = prompt('Show plots', False) if query else False
    sim.postprocess(
        iteration=sim.iteration,
        log_path=log_path if save_data else '',
        plot=show_plots,
        video=(
            os.path.join(log_path, 'simulation.mp4')
            if sim.options.record
            else ''
        ),
    )
    if save_data:
        pylog.debug('Data saved, now loading back to check validity')
        data = AnimatData.from_file(os.path.join(log_path, 'simulation.hdf5'))
        pylog.debug('Data successfully saved and logged back: {}'.format(data))

    # Plot network
    show_connectivity = (
        prompt('Show connectivity maps', False)
        if query
        else False
    )
    if show_connectivity:
        plot_networks_maps(animat_options.morphology, sim.animat().data)

    # Plot
    if (
            (show_plots or show_connectivity)
            and query
            and prompt('Save plots', False)
    ):
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
            and query
            and prompt('Show connectivity plots', False)
    ):
        plt.show()
