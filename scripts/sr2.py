#!/usr/bin/env python3
"""Run salamandra robotica II simulation with bullet"""

import time
import matplotlib.pyplot as plt
from farms_models.utils import get_sdf_path
from farms_amphibious.experiment.simulation import (
    simulation,
    amphibious_options,
    profile,
    get_animat_options,
)
import farms_pylog as pylog


def main():
    """Main"""

    # Animat
    sdf = get_sdf_path(name='salamandra_robotica', version='2')
    pylog.info('Model SDF: {}'.format(sdf))
    animat_options = get_animat_options(
        swimming=False,
        n_legs=4,
        n_dof_legs=1,
        n_joints_body=9,
    )

    (
        simulation_options,
        arena_sdf,
    ) = amphibious_options(animat_options, use_water_arena=False)

    # Simulation
    profile(
        function=simulation,
        animat_sdf=sdf,
        animat_options=animat_options,
        simulation_options=simulation_options,
        arena_sdf=arena_sdf,
        use_controller=True,
    )
    plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
