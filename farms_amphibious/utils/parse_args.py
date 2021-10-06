"""Parse command line arguments"""

import numpy as np
import argparse
from farms_bullet.simulation.parse_args import (
    argument_parser as bullet_argument_parser,
)


def argument_parser():
    """Parse args"""
    parser = bullet_argument_parser()
    parser.description = 'Amphibious simulation'
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
        nargs=3,
        type=float,
        metavar=('x', 'y', 'z'),
        default=(0, 0, 0),
        help='Water velocity',
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
        '--drive_config',
        type=str,
        default='',
        help='Descending drive method',
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
        '--verify_save',
        action='store_true',
        help='Verify if saved simulation data can be loaded',
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
        '--drives',
        nargs=2,
        type=float,
        metavar=('forward', 'turn'),
        default=(2, 0),
        help='Animat descending drives',
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
    parser.add_argument(
        '--prompt',
        action='store_true',
        help='Prompt at end of simulation',
    )
    for weight in [
            'weight_osc_body',
            'weight_osc_legs_internal',
            'weight_osc_legs_opposite',
            'weight_osc_legs_following',
            'weight_osc_legs2body',
            'weight_osc_body2legs',
            'weight_sens_stretch_freq',
            'weight_sens_contact_intralimb',
            'weight_sens_contact_opposite',
            'weight_sens_contact_following',
            'weight_sens_contact_diagonal',
            'weight_sens_hydro_freq',
            'weight_sens_hydro_amp',
    ]:
        parser.add_argument(
            f'--{weight}',
            type=float,
            default=None,
            help=f'{weight}',
        )
    return parser


def parse_args():
    """Parse args"""
    parser = argument_parser()
    return parser.parse_args()


def parse_args_model_gen(description='Generate model'):
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
    return parser.parse_args()
