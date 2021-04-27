"""Generate salamander"""

from farms_amphibious.utils.generation import generate_amphibious


def main():
    """Main"""
    leg_offset = [
        [0, 0.04, -0.02],
        [-0.01, 0.03, -0.02],
    ]
    generate_amphibious(
        model_name='salamander',
        model_version='v3',
        n_joints_body=11,
        leg_length=0.04,
        leg_radius=0.01,
        legs_parents=[1, 5],
        leg_offset=leg_offset,
    )
    generate_amphibious(
        model_name='salamander',
        model_version='v3_short_legs',
        n_joints_body=11,
        leg_length=0.02,
        leg_radius=0.01,
        legs_parents=[1, 5],
        leg_offset=leg_offset,
    )
    generate_amphibious(
        model_name='salamander',
        model_version='v3_long_legs',
        n_joints_body=11,
        leg_length=0.06,
        leg_radius=0.01,
        legs_parents=[1, 5],
        leg_offset=leg_offset,
    )


if __name__ == '__main__':
    main()
