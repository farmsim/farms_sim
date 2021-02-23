"""Generate salamander"""

from farms_amphibious.utils.generation import generate_amphibious


def main():
    """Main"""
    generate_amphibious(
        model_name='salamander',
        model_version='v3',
        n_joints_body=11,
        leg_radius=0.01,
        legs_parents=[1, 4]
    )


if __name__ == '__main__':
    main()
