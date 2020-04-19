"""Utils"""

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
