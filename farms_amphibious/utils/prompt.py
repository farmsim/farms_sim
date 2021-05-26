"""Prompt"""

import os
from distutils.util import strtobool

import matplotlib.pyplot as plt

import farms_pylog as pylog
from farms_models.utils import get_simulation_data_path
from farms_data.amphibious.animat_data import AnimatData
from farms_amphibious.utils.network import plot_networks_maps
from farms_bullet.simulation.parse_args import (
    argument_parser as bullet_argument_parser,
)


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


def argument_parser():
    """Parse args"""
    parser = bullet_argument_parser()
    parser.description = 'Amphibious simulation'
    parser.add_argument(
        '--prompt',
        action='store_true',
        help='Prompt at end of simulation',
    )
    parser.add_argument(
        '--sdf',
        type=str,
        default='',
        help='SDF file',
    )
    parser.add_argument(
        '--animat',
        type=str,
        default='salamander',
        help='Animat',
    )
    parser.add_argument(
        '--version',
        type=str,
        default='',
        help='Animat version',
    )
    parser.add_argument(
        '--position',
        nargs=3,
        type=float,
        metavar=('x', 'y', 'z'),
        default=(0, 0, 0),
        help='Spawn position',
    )
    parser.add_argument(
        '--orientation',
        nargs=3,
        type=float,
        metavar=('alpha', 'beta', 'gamma'),
        default=(0, 0, 0),
        help='Spawn orientation',
    )
    parser.add_argument(
        '--arena',
        type=str,
        choices=('flat', 'ramp', 'water'),  # 'rough', 'pool'
        default='flat',
        help='Simulation arena',
    )
    parser.add_argument(
        '--water',
        type=float,
        default=0.0,
        help='Water surface height',
    )
    parser.add_argument(
        '--ground',
        type=float,
        default=None,
        help='Ground height',
    )
    parser.add_argument(
        '-s', '--save',
        type=str,
        default='',
        help='Save simulation data to provided path',
    )
    parser.add_argument(
        '--save_to_models',
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
    parser.add_argument(
        '--drives',
        nargs=2,
        type=float,
        metavar=('forward', 'turn'),
        default=(2, 0),
        help='Animat descending drives',
    )
    parser.add_argument(
        '--lateral_friction',
        type=float,
        default=1.0,
        help='Lateral friction',
    )
    parser.add_argument(
        '--viscosity',
        type=float,
        default=1.0,
        help='Viscosity',
    )
    parser.add_argument(
        '--self_collisions',
        action='store_true',
        help='Apply self collisions',
    )
    parser.add_argument(
        '--spawn_loader',
        type=str,
        choices=('FARMS', 'PYBULLET'),
        default='FARMS',
        help='Spawn loader',
    )
    return parser


def parse_args():
    """Parse args"""
    parser = argument_parser()
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
    save_to_models = kwargs.pop('save_to_models', '')
    extension = kwargs.pop('extension', 'pdf')

    # Post-processing
    pylog.info('Simulation post-processing')
    log_path = get_simulation_data_path(
        name=animat,
        version=version,
        simulation_name=save if save else 'default',
    ) if save_to_models else save
    save_data = (
        (query and prompt('Save data', False))
        or (save or save_to_models) and not query
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
