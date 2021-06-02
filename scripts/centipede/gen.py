"""Generate centipede"""

from farms_amphibious.utils.generation import generate_amphibious
from farms_amphibious.utils.parse_args import parse_args_model_gen


def main():
    """Main"""
    clargs = parse_args_model_gen(description='Generate centipede')
    kwargs = {
        'model_name': 'centipede',
        'model_version': clargs.version,
        'sdf_path': clargs.sdf_path,
        'model_path': clargs.model_path,
        'body_sdf_path': clargs.original,
        'n_joints_body': 20,
        'legs_parents': list(range(1, 20)),
        'leg_offset': [0.05, 0.06, -0.03],
        'leg_radius': 0.01,
        'eye_pos': [0.015, 0.03, 0.015],
        'eye_radius': 0.02,
        'color': [0.1, 0.0, 0.0, 1.0],
        'scale': 0.05,
    }
    if clargs.version == 'v1':
        generate_amphibious(
            **kwargs,
            leg_length=0.07,
        )
    elif clargs.version == 'v1_short_legs':
        generate_amphibious(
            **kwargs,
            leg_length=0.04,
        )
    elif clargs.version == 'v1_long_legs':
        generate_amphibious(
            **kwargs,
            leg_length=0.10,
        )
    else:
        raise Exception('Unknown model: {}'.format(clargs))


if __name__ == '__main__':
    main()
