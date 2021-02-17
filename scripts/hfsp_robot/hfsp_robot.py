#!/usr/bin/env python3
"""Run hfsp_robot simulation with bullet"""

import os
import time
import matplotlib.pyplot as plt

import farms_pylog as pylog
from farms_bullet.utils.profile import profile
from farms_bullet.simulation.options import SimulationOptions
from farms_amphibious.utils.network import plot_networks_maps
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.experiment.simulation import simulation
from farms_amphibious.experiment.options import (
    get_hfsp_robot_options,
    amphibious_options,
)
from farms_amphibious.utils.prompt import (
    parse_args,
    prompt_postprocessing,
)


def main():
    """Main"""

    # Arguments
    clargs = parse_args()

    # Options
    sdf, animat_options = get_hfsp_robot_options(
        drives_init=clargs.drives,
        spawn_position=clargs.position,
        spawn_orientation=clargs.orientation,
    )
    simulation_options, arena = amphibious_options(
        animat_options=animat_options,
        arena=clargs.arena,
    )

    # # State
    # n_joints = animat_options.morphology.n_joints()
    # state_init = (1e-3*np.arange(5*n_joints)).tolist()
    # for osc_i, osc in enumerate(animat_options.control.network.oscillators):
    #     osc.initial_phase = state_init[osc_i]
    #     osc.initial_amplitude = state_init[osc_i+n_joints]

    if clargs.test:
        # Save options
        animat_options_filename = 'hfsp_robot_animat_options.yaml'
        animat_options.save(animat_options_filename)
        simulation_options_filename = 'hfsp_robot_simulation_options.yaml'
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
        animat='hfsp_robot',
        version='0',
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
