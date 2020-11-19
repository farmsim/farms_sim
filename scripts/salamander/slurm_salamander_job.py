#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import os
import time

import farms_pylog as pylog
from farms_models.utils import get_sdf_path
from farms_bullet.simulation.options import SimulationOptions
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.experiment.simulation import simulation
from farms_amphibious.experiment.options import (
    amphibious_options,
    get_salamander_options,
)


def main():
    """Main"""

    # Animat
    sdf = get_sdf_path(name='salamander', version='v1')
    pylog.info('Model SDF: {}'.format(sdf))
    animat_options = get_salamander_options(
        # spawn_position=[-5, 0, 0.1],
        # spawn_orientation=[0, 0, np.pi],
        # drives_init=[4.9, 0],
    )

    simulation_options, arena = amphibious_options(
        animat_options,
        use_water_arena=True,
    )

    # Save options
    animat_options_filename = 'salamander_animat_options.yaml'
    animat_options.save(animat_options_filename)
    simulation_options_filename = 'salamander_simulation_options.yaml'
    simulation_options.save(simulation_options_filename)

    # Load options
    animat_options = AmphibiousOptions.load(animat_options_filename)
    simulation_options = SimulationOptions.load(simulation_options_filename)

    # Simulation
    sim = simulation(
        animat_sdf=sdf,
        animat_options=animat_options,
        simulation_options=simulation_options,
        arena=arena,
        use_controller=True,
    )


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
