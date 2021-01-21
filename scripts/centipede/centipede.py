#!/usr/bin/env python3
"""Run centipede simulation with bullet"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import farms_pylog as pylog
from farms_models.utils import get_sdf_path, get_simulation_data_path
from farms_data.amphibious.animat_data import AnimatData
from farms_bullet.utils.profile import profile
from farms_bullet.model.options import SpawnLoader
from farms_bullet.model.control import ControlType
from farms_bullet.simulation.options import SimulationOptions
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.utils.prompt import prompt
from farms_amphibious.utils.network import plot_networks_maps
from farms_amphibious.experiment.simulation import simulation
from farms_amphibious.experiment.options import (
    amphibious_options,
    get_centipede_options,
)
from farms_amphibious.utils.prompt import (
    parse_args,
    prompt_postprocessing,
)


def main(animat='centipede', version='v1', scale=0.2):
    """Main"""

    # Animat
    sdf = get_sdf_path(name=animat, version=version)
    pylog.info('Model SDF: {}'.format(sdf))
    animat_options = get_centipede_options(
        spawn_loader=SpawnLoader.PYBULLET,
    )

    simulation_options, arena = amphibious_options(
        animat_options,
        use_water_arena=False,
    )
    simulation_options.units.meters = 10
    simulation_options.units.seconds = 1
    simulation_options.units.kilograms = 1
    animat_options.show_hydrodynamics = not simulation_options.headless

    # Save options
    animat_options_filename = 'centipede_animat_options.yaml'
    animat_options.save(animat_options_filename)
    simulation_options_filename = 'centipede_simulation_options.yaml'
    simulation_options.save(simulation_options_filename)

    # Load options
    animat_options = AmphibiousOptions.load(animat_options_filename)
    simulation_options = SimulationOptions.load(simulation_options_filename)

    # Simulation
    sim = profile(
        function=simulation,
        animat_sdf=sdf,
        animat_options=animat_options,
        simulation_options=simulation_options,
        arena=arena,
        use_controller=True,
    )

    # Post-processing
    if parse_args()[0].prompt:
        prompt_postprocessing(
            animat=animat,
            version=version,
            sim=sim,
            animat_options=animat_options,
        )


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
