#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import os
import time
import matplotlib.pyplot as plt

import farms_pylog as pylog
from farms_models.utils import get_sdf_path
from farms_bullet.simulation.options import SimulationOptions
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.experiment.simulation import simulation, profile
from farms_amphibious.experiment.options import (
    amphibious_options,
    get_animat_options,
)


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
            [-1e-1, -1e1, -1e1],
            [-1e-6, -1e-6, -1e-6],
        ],
        weight_osc_body=1e0,
        weight_osc_legs_internal=3e1,
        weight_osc_legs_opposite=3e0,
        weight_osc_legs_following=3e0,
        weight_osc_legs2body=1e1,
        weight_sens_contact_i=-2e0,
        weight_sens_contact_e=2e0,
        weight_sens_hydro_freq=-1e-1,
        weight_sens_hydro_amp=-1e-1,
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
    sim = profile(
        function=simulation,
        animat_sdf=sdf,
        animat_options=animat_options,
        simulation_options=simulation_options,
        arena_sdf=arena_sdf,
        use_controller=True,
    )

    # Post-processing
    postprocessing = False
    if postprocessing:
        pylog.info('Simulation post-processing')
        video_name = ''
        log_path = 'salamander_results'
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        sim.postprocess(
            iteration=sim.iteration,
            log_path=log_path,
            plot=True,
            video=video_name if not sim.options.headless else ''
        )

    # Plot
    plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
