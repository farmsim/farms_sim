"""Generate salamander"""

from farms_amphibious.utils.generation import generate_amphibious
from farms_amphibious.utils.parse_args import parse_args_model_gen


def main():
    """Main"""
    clargs = parse_args_model_gen(description='Generate salamander')
    kwargs = {
        'model_name': 'salamander',
        'model_version': clargs.version,
        'sdf_path': clargs.sdf_path,
        'model_path': clargs.model_path,
        'body_sdf_path': clargs.original,
        'n_joints_body': 11,
        'legs_parents': [1, 5],
        'leg_offset': [[0, 0.04, -0.02], [-0.01, 0.03, -0.02]],
        'leg_radius': 0.01,
    }
    if clargs.version == 'v3':
        generate_amphibious(
            **kwargs,
            leg_length=0.04,
        )
    elif clargs.version == 'v3_short_legs':
        generate_amphibious(
            **kwargs,
            leg_length=0.02,
        )
    elif clargs.version == 'v3_long_legs':
        generate_amphibious(
            **kwargs,
            leg_length=0.06,
        )
    else:
        raise Exception('Unknown model: {}'.format(clargs))


if __name__ == '__main__':
    main()
