"""Parse command line arguments"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace


def sim_argument_parser() -> ArgumentParser:
    """Argument parser"""
    parser = ArgumentParser(
        description='FARMS simulation',
        formatter_class=(
            lambda prog:
            ArgumentDefaultsHelpFormatter(prog, max_help_position=50)
        ),
    )

    # Simulator
    parser.add_argument(
        '--simulator',
        type=str,
        choices=('MUJOCO', 'PYBULLET'),
        default='MUJOCO',
        help='Simulator',
    )

    # Experiment config files
    parser.add_argument(
        '--simulation_config',
        type=str,
        default=None,
        help='Simulation config',
    )
    parser.add_argument(
        '--animat_config',
        type=str,
        default=None,
        help='Animat config',
    )
    parser.add_argument(
        '--arena_config',
        type=str,
        default=None,
        help='Arena config',
    )

    # Profiling
    parser.add_argument(
        '--profile',
        type=str,
        default='',
        help='Save simulation profile to given filename',
    )

    # Logging
    parser.add_argument(
        '--log_path',
        type=str,
        default='',
        help='Log simulation data to provided folder path',
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


def sim_parse_args() -> Namespace:
    """Parse arguments"""
    parser = sim_argument_parser()
    # return parser.parse_args()
    args, _ = parser.parse_known_args()
    return args
