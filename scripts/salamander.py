#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time
import matplotlib.pyplot as plt
from farms_models.utils import get_sdf_path
from farms_amphibious.examples.simulation import (
    simulation,
    profile,
    get_animat_options,
    get_simulation_options,
)
import farms_pylog as pylog


def main():
    """Main"""
    sdf = get_sdf_path(name='salamander', version='v1')
    pylog.info('Model SDF: {}'.format(sdf))
    animat_options = get_animat_options(swimming=False)
    animat_options.morphology.n_legs = 4
    animat_options.morphology.n_dof_legs = 4
    animat_options.morphology.n_joints_body = 11
    simulation_options = get_simulation_options()
    profile(
        function=simulation,
        sdf=sdf,
        animat_options=animat_options,
        simulation_options=simulation_options,
        use_controller=True,
        water_arena=False
    )
    plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
