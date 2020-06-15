#!/usr/bin/env python3
"""Run fish simulation with bullet"""

import os
import time
import matplotlib.pyplot as plt
from farms_models.utils import (
    get_sdf_path,
    model_kinematics_files,
    get_simulation_data_path,
)
import farms_pylog as pylog
from farms_amphibious.experiment.simulation import simulation, profile
from farms_amphibious.experiment.options import (
    get_fish_options,
    get_fish_kwargs_options,
)


def main():
    """Main"""
    fish_name = 'crescent_gunnel'
    fish_version = '1'
    animat_sdf = get_sdf_path(
        name=fish_name,
        version=fish_version
    )
    pylog.info('Model SDF: {}'.format(animat_sdf))
    default_options = get_fish_kwargs_options()
    for kinematics_file in model_kinematics_files(fish_name, fish_version):
        # Get options
        (
            animat_options,
            arena,
            simulation_options,
            _kinematics,
        ) = get_fish_options(
            fish_name,
            fish_version,
            kinematics_file,
            **default_options
        )

        # Simulation
        sim = profile(
            function=simulation,
            animat_sdf=animat_sdf,
            animat_options=animat_options,
            simulation_options=simulation_options,
            use_controller=True,
            sampling=default_options['sampling_timestep'],
            arena=arena,
        )

        # Post-processing
        pylog.info('Simulation post-processing')
        log_path = get_simulation_data_path(
            name=fish_name,
            version=fish_version,
            simulation_name=os.path.basename(kinematics_file),
        )
        video_name = ''
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        sim.postprocess(
            iteration=sim.iteration,
            log_path=log_path,
            plot=False,
            video=video_name if not sim.options.headless else ''
        )
        plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
