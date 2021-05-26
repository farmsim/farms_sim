"""Parse command line arguments"""

from farms_bullet.simulation.parse_args import (
    argument_parser as bullet_argument_parser,
)


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
