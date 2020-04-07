#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time
import matplotlib.pyplot as plt
from farms_models.utils import get_sdf_path
from farms_bullet.simulation.options import SimulationOptions
from farms_amphibious.model.options import AmphibiousOptions
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
        show_hydrodynamics=True,
        swimming=False,
        n_legs=4,
        n_dof_legs=4,
        n_joints_body=11,
        viscous_coefficients=[
            [-1e-1, -1e0, -1e0],
            [-1e-6, -1e-6, -1e-6],
        ]
    )

    (
        simulation_options,
        arena_sdf,
    ) = amphibious_options(animat_options, use_water_arena=True)

    # Save options
    animat_options_filename = 'salamander_animat_options.yaml'
    animat_options.save(animat_options_filename)
    simulation_options_filename = 'salamander_simulation_options.yaml'
    simulation_options.save(simulation_options_filename)

    # Load options
    animat_options = AmphibiousOptions.load(animat_options_filename)
    simulation_options = SimulationOptions.load(simulation_options_filename)

    # Simulation
    profile(
        function=simulation,
        animat_sdf=sdf,
        animat_options=animat_options,
        simulation_options=simulation_options,
        arena_sdf=arena_sdf,
        use_controller=True,
    )

    # Plot
    plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
