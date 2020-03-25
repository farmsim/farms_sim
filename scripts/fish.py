#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time
import numpy as np
import matplotlib.pyplot as plt
from farms_models.utils import get_sdf_path, model_kinematics_files
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.examples.simulation import (
    simulation,
    profile,
    # get_animat_options,
    get_flat_arena,
    get_simulation_options,
)
import farms_pylog as pylog


def main():
    """Main"""
    fish_name = 'crescent_gunnel'
    fish_version = '0'
    animat_sdf = get_sdf_path(
        name=fish_name,
        version=fish_version
    )
    pylog.info('Model SDF: {}'.format(animat_sdf))
    n_joints = 20
    animat_options = AmphibiousOptions(
        show_hydrodynamics=True,
        n_joints_body=n_joints,
        viscous=False,
        resistive=True,
        resistive_coefficients=[
            1e-1*np.array([-1e-4, -5e-1, -3e-1]),
            1e-1*np.array([-1e-6, -1e-6, -1e-6])
        ],
        water_surface=False
    )
    animat_options.morphology.n_legs = 0
    animat_options.morphology.n_dof_legs = 0

    # Arena
    arena_sdf = get_flat_arena()

    # get_animat_options(swimming=False)
    simulation_options = get_simulation_options()
    simulation_options.gravity = [0, 0, 0]
    simulation_options.timestep = 1e-3
    simulation_options.units.meters = 1
    simulation_options.units.seconds = 1e3
    simulation_options.units.kilograms = 1

    # Logging
    simulation_options.log_path = "fish_results"

    # Camera options
    simulation_options.video_yaw = 0
    simulation_options.video_pitch = -30
    simulation_options.video_distance = 1
    # simulation_options.video_name = (
    #     "transition_videos/swim2walk_y{}_p{}_d{}".format(
    #         simulation_options.video_yaw,
    #         simulation_options.video_pitch,
    #         simulation_options.video_distance,
    #     )
    # )

    n_sample = 100
    sampling_timestep = 1e-2
    for kinematics_file in model_kinematics_files(fish_name, fish_version):

        # Kinematics data handling
        pylog.info(kinematics_file)
        animat_options.control.kinematics_file = kinematics_file
        kinematics = np.loadtxt(animat_options.control.kinematics_file)
        len_kinematics = np.shape(kinematics)[0]
        simulation_options.duration = len_kinematics*sampling_timestep
        pose = kinematics[:, :3]
        position = np.ones(3)
        position[:2] = pose[0, :2]
        orientation = np.zeros(3)
        orientation[2] = pose[0, 2]
        velocity = np.zeros(3)
        velocity[:2] = pose[n_sample, :2] - pose[0, :2]
        velocity /= n_sample*sampling_timestep
        kinematics = kinematics[:, 3:]
        kinematics = ((kinematics + np.pi) % (2*np.pi)) - np.pi

        # Animat options
        animat_options.spawn.position = position
        animat_options.spawn.orientation = orientation
        animat_options.physics.buoyancy = False
        animat_options.spawn.velocity_lin = velocity
        animat_options.spawn.velocity_ang = [0, 0, 0]
        animat_options.spawn.joints_positions = kinematics[0, :]
        animat_options.morphology.n_joints_body = n_joints
        # np.shape(kinematics)[1] - 3
        # animat_options.spawn.position = [-10, 0, 0]
        # animat_options.spawn.orientation = [0, 0, np.pi]

        # Simulation
        profile(
            function=simulation,
            animat_sdf=animat_sdf,
            animat_options=animat_options,
            simulation_options=simulation_options,
            use_controller=True,
            arena_sdf=arena_sdf,
            links=['link_body_0']+[
                'body_{}_t_link'.format(i+1)
                for i in range(n_joints-1)
            ],
            joints=['joint_{}'.format(i) for i in range(n_joints)],
        )
        plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
