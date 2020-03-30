#!/usr/bin/env python3
"""Run fish simulation with bullet"""

import time
import numpy as np
import matplotlib.pyplot as plt
from farms_models.utils import get_sdf_path, model_kinematics_files
from farms_amphibious.experiment.simulation import (
    simulation,
    profile,
    fish_options,
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
    sampling_timestep = 1.2e-2
    for kinematics_file in model_kinematics_files(fish_name, fish_version):
        # Get options
        (
            animat_options,
            arena_sdf,
            simulation_options,
            _kinematics,
            links,
            joints,
        ) = fish_options(
            kinematics_file,
            sampling_timestep,
            resistive_coefficients=[
                1e-1*np.array([-1e-4, -5e-1, -3e-1]),
                1e-1*np.array([-1e-6, -1e-6, -1e-6]),
            ]
        )

        # Logging
        simulation_options.log_path = "fish_results"

        # Simulation
        profile(
            function=simulation,
            animat_sdf=animat_sdf,
            animat_options=animat_options,
            simulation_options=simulation_options,
            use_controller=True,
            sampling=sampling_timestep,
            arena_sdf=arena_sdf,
            links=links,
            links_swimming=links,
            joints=joints,
        )
        plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
