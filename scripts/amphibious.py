#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time
# import numpy as np

import farms_pylog as pylog
from farms_models.utils import get_sdf_path
from farms_bullet.utils.profile import profile
from farms_bullet.model.options import SpawnLoader
from farms_bullet.model.control import ControlType
from farms_bullet.simulation.options import SimulationOptions
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.utils.prompt import (
    parse_args,
    prompt_postprocessing,
)

from farms_amphibious.experiment.simulation import simulation
from farms_amphibious.experiment.options import (
    amphibious_options,
    get_animat_options_from_model,
)


def main():
    """Main"""

    # Arguments
    clargs = parse_args()

    # Options
    sdf = get_sdf_path(name=clargs.animat, version=clargs.version)
    options = {
        'drives_init': clargs.drives,
        'spawn_position': clargs.position,
        'spawn_orientation': clargs.orientation,
        'spawn_loader': SpawnLoader.PYBULLET,
        'default_control_type': ControlType.POSITION,
    }
    animat_options = get_animat_options_from_model(
        animat=clargs.animat,
        version=clargs.version,
        default_lateral_friction=clargs.lateral_friction,
        **options,
    )
    simulation_options, arena = amphibious_options(
        animat_options=animat_options,
        arena=clargs.arena,
        viscosity=clargs.viscosity,
        water_surface=clargs.water,
        ground_height=clargs.ground,
    )

    # Test options saving and loading
    if clargs.test:
        # Save options
        animat_options_filename = 'salamander_animat_options.yaml'
        animat_options.save(animat_options_filename)
        simulation_options_filename = 'salamander_simulation_options.yaml'
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
        profile_filename=clargs.profile,
    )

    # Post-processing
    prompt_postprocessing(
        animat=clargs.animat,
        version=clargs.version,
        sim=sim,
        animat_options=animat_options,
        query=clargs.prompt,
        save=clargs.save,
        models=clargs.models,
    )


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
