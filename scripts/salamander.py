#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time
# import yaml
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
    sdf = get_sdf_path(name='salamander', version='v1')
    pylog.info('Model SDF: {}'.format(sdf))
    animat_options = get_animat_options(
        swimming=False,
        n_legs=4,
        n_dof_legs=4,
        n_joints_body=11,
    )

    (
        simulation_options,
        arena_sdf,
    ) = amphibious_options(animat_options, use_water_arena=True)

    # # Save options
    # with open('animat_options.yaml', 'w+') as yaml_file:
    #     yaml.dump(
    #         animat_options,
    #         yaml_file,
    #         default_flow_style=False
    #     )

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
