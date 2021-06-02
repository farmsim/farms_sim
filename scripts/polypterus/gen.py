"""Generate polypterus"""

from farms_amphibious.utils.generation import generate_amphibious
from farms_amphibious.utils.parse_args import parse_args_model_gen


def main():
    """Main"""
    clargs = parse_args_model_gen(description='Generate polypterus')
    kwargs = {
        'model_name': 'polypterus',
        'model_version': clargs.version,
        'sdf_path': clargs.sdf_path,
        'model_path': clargs.model_path,
        'body_sdf_path': clargs.original,
        'n_joints_body': 20,
        'legs_parents': [1],
        'leg_offset': [0, 0.06, -0.02],
        'leg_radius': 0.01,
        'eye_pos': [0.015, 0.03, 0.015],
        'eye_radius': 0.02,
        'color': [1.0, 1.0, 0.0, 1.0],
        'scale': 0.05,
    }
    if clargs.version == 'v0':
        generate_amphibious(
            **kwargs,
            leg_length=0.04,
        )
    elif clargs.version == 'v0_short_fins':
        generate_amphibious(
            **kwargs,
            leg_length=0.02,
        )
    elif clargs.version == 'v0_long_fins':
        generate_amphibious(
            **kwargs,
            leg_length=0.06,
        )
    else:
        raise Exception('Unknown model: {}'.format(clargs))


if __name__ == '__main__':
    main()
