"""Prompt"""

import argparse
from distutils.util import strtobool


def prompt(query, default):
    """Prompt"""
    val = input('{} [{}]: '.format(
        query,
        'Y/n' if default else 'y/N',
    ))
    try:
        ret = strtobool(val) if val is not '' else default
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
