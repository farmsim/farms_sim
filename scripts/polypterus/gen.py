"""Generate polypterus"""

from farms_amphibious.utils.generation import generate_amphibious

def main():
    """Main"""
    generate_amphibious(
        model_name='polypterus',
        model_version='v0',
        n_joints_body=20,
        scale=0.05,
        leg_offset=[0, 0.06, -0.01],
        leg_length=0.04,
        leg_radius=0.01,
        legs_parents=[1],
        eye_pos=[0.015, 0.03, 0.015],
        eye_radius=0.02,
        color=[1.0, 1.0, 0.0, 1.0],
    )


if __name__ == '__main__':
    main()
