"""Generate centipede"""

from farms_amphibious.utils.generation import generate_amphibious

def main():
    """Main"""
    generate_amphibious(
        model_name='centipede',
        model_version='v1',
        n_joints_body=20,
        scale=0.05,
        leg_offset=[0.05, 0.06, -0.03],
        leg_length=0.07,
        leg_radius=0.01,
        legs_parents=list(range(1, 20)),
        eye_pos=[0.015, 0.03, 0.015],
        eye_radius=0.02,
        color=[0.1, 0.0, 0.0, 1.0],
    )


if __name__ == '__main__':
    main()
