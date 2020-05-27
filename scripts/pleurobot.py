#!/usr/bin/env python3
"""Run pleurobot simulation with bullet"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import farms_pylog as pylog
from farms_bullet.model.control import ControlType
from farms_bullet.simulation.options import SimulationOptions
from farms_amphibious.utils.utils import prompt
from farms_amphibious.utils.network import plot_networks_maps
from farms_amphibious.experiment.simulation import simulation, profile
from farms_amphibious.model.options import AmphibiousOptions, SpawnLoader
from farms_amphibious.experiment.options import (
    get_pleurobot_options,
    amphibious_options,
)


def main():
    """Main"""

    sdf, animat_options = get_pleurobot_options(
        spawn_loader=SpawnLoader.PYBULLET,  # SpawnLoader.FARMS
        default_control_type=ControlType.POSITION,
        weight_osc_body=1e0,
        weight_osc_legs_internal=3e1,
        weight_osc_legs_opposite=1e-1,  # 1e1,
        weight_osc_legs_following=0,  # 1e1,
        weight_osc_legs2body=3e1,
        weight_sens_contact_intralimb=-2e-1,
        weight_sens_contact_opposite=5e-1,
        weight_sens_contact_following=0,
        weight_sens_contact_diagonal=0,
        weight_sens_hydro_freq=0,
        weight_sens_hydro_amp=0,
        body_stand_amplitude=0.2,
        modular_phases=np.array([3*np.pi/2, 0, 3*np.pi/2, 0]) - np.pi/4,
        modular_amplitudes=np.full(4, 0.9),
        legs_amplitudes=[np.pi/8, np.pi/16, np.pi/8, np.pi/8],
        default_lateral_friction=2,
    )

    # # State
    # n_joints = animat_options.morphology.n_joints()
    # state_init = (1e-3*np.arange(5*n_joints)).tolist()
    # for osc_i, osc in enumerate(animat_options.control.network.oscillators):
    #     osc.initial_phase = state_init[osc_i]
    #     osc.initial_amplitude = state_init[osc_i+n_joints]

    # Muscles
    for muscle in animat_options.control.muscles:
        muscle.alpha = 5e0
        muscle.beta = -3e0
        muscle.gamma = 3e0
        muscle.delta = -2e-3

    (
        simulation_options,
        arena,
    ) = amphibious_options(animat_options, use_water_arena=False)

    # Save options
    animat_options_filename = 'pleurobot_animat_options.yaml'
    animat_options.save(animat_options_filename)
    simulation_options_filename = 'pleurobot_simulation_options.yaml'
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
    pylog.info('Simulation post-processing')
    log_path = 'pleurobot_results'
    video_name = os.path.join(log_path, 'simulation.mp4')
    if log_path and not os.path.isdir(log_path):
        os.mkdir(log_path)
    sim.postprocess(
        iteration=sim.iteration,
        log_path=log_path if prompt('Save data', False) else '',
        plot=prompt('Show plots', False),
        video=video_name if sim.options.record else ''
    )

    # Plot network
    if prompt('Show connectivity maps', False):
        plot_networks_maps(animat_options.morphology, sim.animat().data)

    # Plot
    plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
