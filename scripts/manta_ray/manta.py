"""Manta"""

import os

from farms_sdf.sdf import ModelSDF
from farms_data.amphibious.animat_data import AnimatData
from farms_models.utils import get_sdf_path, get_simulation_data_path
from farms_bullet.model.control import ControlType
from farms_bullet.utils.profile import profile
from farms_bullet.model.options import (
    # ModelOptions,
    # MorphologyOptions,
    # LinkOptions,
    JointOptions,
    SpawnOptions,
    # ControlOptions,
    # JointControlOptions,
    # SensorsOptions,
)

from farms_amphibious.experiment.simulation import simulation
from farms_amphibious.control.manta_control import joints_sorted, control
from farms_amphibious.model.options import (
    AmphibiousOptions,
    AmphibiousPhysicsOptions,
    AmphibiousMorphologyOptions,
    AmphibiousLinkOptions,
    # AmphibiousJointOptions,
    # AmphibiousSpawnOptions,
    AmphibiousControlOptions,
    AmphibiousJointControlOptions,
    AmphibiousSensorsOptions,
    # AmphibiousNetworkOptions,
)
from farms_amphibious.experiment.options import (
    get_simulation_options,
    get_flat_arena,
)


def manta_options(sdf, drag=-1e-3):
    """Manta options"""
    model = ModelSDF.read(sdf)[0]
    links_names = [link.name for link in model.links]
    joints_names = [joint.name for joint in model.joints]
    print('Number of links: {}'.format(len(links_names)))
    print('Number of joints: {}'.format(len(joints_names)))
    (
        j_left,
        j_right,
        j_passive_left,
        j_passive_right,
    ) = joints_sorted(names=joints_names)
    j_control = control(0, j_left, j_right, j_passive_left, j_passive_right)
    return AmphibiousOptions(
        spawn=SpawnOptions.from_options({}),
        morphology=AmphibiousMorphologyOptions(
            links=[
                AmphibiousLinkOptions(
                    name=link_name,
                    collisions=False,
                    density=1000.0,
                    swimming=True,
                    drag_coefficients=[
                        [drag, drag, drag],
                        [0.0, 0.0, 0.0],
                    ],
                    mass_multiplier=1.0,
                    pybullet_dynamics={},
                )
                for link_name in links_names
            ],
            self_collisions=[],
            joints=[
                JointOptions(
                    name=joint_name,
                    initial_position=j_control[joint_name],
                    initial_velocity=0,
                    pybullet_dynamics={},
                )
                for joint_name in joints_names
            ],
            n_joints_body=0,
            n_dof_legs=0,
            n_legs=0,
        ),
        control=AmphibiousControlOptions(
            sensors=AmphibiousSensorsOptions(
                links=links_names,
                joints=joints_names,
                contacts=[],
                hydrodynamics=links_names,
            ),
            joints=[
                AmphibiousJointControlOptions(
                    joint=name,
                    control_type=ControlType.POSITION,
                    max_torque=10,
                    offset_gain=0,
                    offset_bias=0,
                    offset_low=0,
                    offset_high=0,
                    offset_saturation=0,
                    rate=0,
                    gain_amplitude=1,
                    bias=0,
                )
                for name in joints_names
            ],
            kinematics_file=0,
            manta_controller=True,
            kinematics_sampling=0,
            network=None,
            muscles=[],
        ),
        physics=AmphibiousPhysicsOptions(
            drag=True,
            sph=False,
            buoyancy=False,
            water_surface=10,
        ),
        show_hydrodynamics=True,
    )


def main():
    """Main"""
    # models
    sdf = get_sdf_path(name='manta_ray', version='0')
    arena = get_flat_arena()

    # Options
    animat_options = manta_options(sdf)
    sim_options = get_simulation_options()
    sim_options.gravity = [0, 0, 0]

    # Simulation
    sim = profile(
        function=simulation,
        animat_sdf=sdf,
        animat_options=animat_options,
        simulation_options=sim_options,
        arena=arena,
        use_controller=True,
    )

    # Post-processing
    log_path = get_simulation_data_path(
        name='manta_ray',
        version='0',
        simulation_name='default',
    )
    if log_path and not os.path.isdir(log_path):
        os.mkdir(log_path)
    sim.postprocess(
        iteration=sim.iteration,
        log_path=log_path,
        plot=False,
        video='',
    )
    AnimatData.from_file(os.path.join(log_path, 'simulation.hdf5'))
    return sim


if __name__ == '__main__':
    main()
