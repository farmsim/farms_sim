"""Parse command line arguments"""

import argparse
import numpy as np
from farms_data.simulation.parse_args import (
    sim_argument_parser as farms_data_sim_argument_parser,
    config_argument_parser as farms_data_config_argument_parser,
)
from ..model.options import options_kwargs_keys


def sim_argument_parser() -> argparse.ArgumentParser:
    """Argument parser"""
    parser = farms_data_sim_argument_parser()
    parser.description = 'FARMS amphibious simulation'

    # Logging and profiling
    parser.add_argument(
        '--log_path',
        type=str,
        default='',
        help='Log simulation data to provided folder path',
    )
    parser.add_argument(
        '--profile',
        type=str,
        default='',
        help='Save simulation profile to given filename',
    )
    parser.add_argument(
        '--test_configs',
        action='store_true',
        help='Test simulation configuration files',
    )
    parser.add_argument(
        '--prompt',
        action='store_true',
        help='Prompt at end of simulation',
    )
    parser.add_argument(
        '--verify_save',
        action='store_true',
        help='Verify if saved simulation data can be loaded',
    )

    return parser


def sim_parse_args():
    """Parse arguments"""
    parser = sim_argument_parser()
    # return parser.parse_args()
    args, _ = parser.parse_known_args()
    return args


def config_argument_parser() -> argparse.ArgumentParser:
    """Parse args"""
    parser = farms_data_config_argument_parser()
    parser.description = 'Amphibious simulation config generation'
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
        default='',
        help='Simulation arena',
    )
    parser.add_argument(
        '--arena_sdf',
        type=str,
        default='',
        help='Arena SDF file',
    )
    parser.add_argument(
        '--arena_position',
        nargs=3,
        type=float,
        metavar=('x', 'y', 'z'),
        default=(0, 0, 0),
        help='Arena position',
    )
    parser.add_argument(
        '--arena_orientation',
        nargs=3,
        type=float,
        metavar=('alpha', 'beta', 'gamma'),
        default=(0, 0, 0),
        help='Arena orientation',
    )
    parser.add_argument(
        '--water_height',
        type=float,
        default=None,
        help='Water surface height',
    )
    parser.add_argument(
        '--water_sdf',
        type=str,
        default='',
        help='Water SDF file',
    )
    parser.add_argument(
        '--water_velocity',
        nargs='+',
        type=float,
        default=(0, 0, 0),
        help='Water velocity (For a constant flow, just provide (vx, vy, vz))',
    )
    parser.add_argument(
        '--water_maps',
        nargs=2,
        type=str,
        metavar=('png_vx', 'png_vy'),
        default=['', ''],
        help='Water maps',
    )
    parser.add_argument(
        '--ground_height',
        type=float,
        default=None,
        help='Ground height',
    )
    parser.add_argument(
        '--control_type',
        type=str,
        default='position',
        help='Control type',
    )
    parser.add_argument(
        '--torque_equation',
        type=str,
        default=None,
        help='Torque equation',
    )
    parser.add_argument(
        '--save_to_models',
        action='store_true',
        help='Save data to farms_models_data',
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
        '--drive_config',
        type=str,
        default='',
        help='Descending drive config',
    )
    parser.add_argument(
        '--max_torque',
        type=float,
        default=np.inf,
        help='Max torque',
    )
    parser.add_argument(
        '--max_velocity',
        type=float,
        default=np.inf,
        help='Max velocity',
    )
    parser.add_argument(
        '--lateral_friction',
        type=float,
        default=1.0,
        help='Lateral friction',
    )
    parser.add_argument(
        '--feet_friction',
        nargs='+',
        type=float,
        default=None,
        help='Feet friction',
    )
    parser.add_argument(
        '--default_restitution',
        type=float,
        default=0.0,
        help='Default restitution',
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
    for key in options_kwargs_keys():
        parser.add_argument(
            f'--{key}',
            type=float,
            default=None,
            help=f'{key}',
        )
    parser.add_argument(
        '--simulator',
        type=str,
        choices=('MUJOCO', 'PYBULLET'),
        default='MUJOCO',
        help='Simulator',
    )
    return parser


def config_parse_args():
    """Parse args"""
    parser = config_argument_parser()
    return parser.parse_args()


def parser_model_gen(description='Generate model'):
    """Parse args"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--animat',
        type=str,
        default='',
        help='Animat name',
    )
    parser.add_argument(
        '--version',
        type=str,
        default='',
        help='Animat version',
    )
    parser.add_argument(
        '--sdf_path',
        type=str,
        default='',
        help='SDF path',
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='',
        help='Model directory path',
    )
    parser.add_argument(
        '--original',
        type=str,
        default='',
        help='Original file',
    )
    return parser


def parse_args_model_gen(*args, **kwargs):
    """Parse args"""
    parser = parser_model_gen(*args, **kwargs)
    return parser.parse_args()
