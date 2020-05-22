#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import farms_pylog as pylog
from farms_models.utils import get_sdf_path
from farms_bullet.simulation.options import SimulationOptions
from farms_amphibious.model.options import AmphibiousOptions, SpawnLoader
from farms_amphibious.utils.utils import prompt
from farms_amphibious.utils.network import plot_networks_maps
from farms_amphibious.experiment.simulation import simulation, profile
from farms_amphibious.data.animat_data import AnimatData
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
        spawn_loader=SpawnLoader.FARMS,  # SpawnLoader.PYBULLET
        show_hydrodynamics=True,
        swimming=False,
        n_legs=4,
        n_dof_legs=4,
        n_joints_body=11,
        drag_coefficients=[
            [-1e-1, -1e1, -1e1],
            [-1e-6, -1e-6, -1e-6],
        ],
        weight_osc_body=1e0,
        weight_osc_legs_internal=3e1,
        weight_osc_legs_opposite=1e0,  # 1e1,
        weight_osc_legs_following=0,  # 1e1,
        weight_osc_legs2body=3e1,
        weight_sens_contact_intralimb=-0.5,
        weight_sens_contact_opposite=2,
        weight_sens_contact_following=0,
        weight_sens_contact_diagonal=0,
        weight_sens_hydro_freq=-1e-1,
        weight_sens_hydro_amp=-1e-1,
        body_stand_amplitude=0.2,
        modular_phases=np.array([3*np.pi/2, 0, 3*np.pi/2, 0]) - np.pi/4,
        modular_amplitudes=np.full(4, 1.0),
    )
    # state_init = animat_options.control.network.state_init
    # for phase_i, phase in enumerate(np.linspace(2*np.pi, 0, 11)):
    #     state_init[2*phase_i] = float(phase)
    #     state_init[2*phase_i+1] = float(phase)+np.pi
    # state_init = animat_options.control.network.state_init
    # for osc_i in range(4*animat_options.morphology.n_joints()):
    #     state_init[osc_i] = 1e-4*np.random.ranf()
    n_joints = animat_options.morphology.n_joints()
    state_init = (1e-4*np.random.ranf(5*n_joints)).tolist()
    for osc_i, osc in enumerate(animat_options.control.network.oscillators):
        osc.initial_phase = state_init[osc_i]
        osc.initial_amplitude = state_init[osc_i+n_joints]

    (
        simulation_options,
        arena,
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
        arena=arena,
        use_controller=True,
    )

    # Post-processing
    pylog.info('Simulation post-processing')
    log_path = 'salamander_results'
    video_name = os.path.join(log_path, 'simulation.mp4')
    save_data = prompt('Save data', False)
    if log_path and not os.path.isdir(log_path):
        os.mkdir(log_path)
    sim.postprocess(
        iteration=sim.iteration,
        log_path=log_path if save_data else '',
        plot=prompt('Show plots', False),
        video=video_name if sim.options.record else ''
    )
    if save_data:
        pylog.debug('Data saved, now loading back to check validity')
        data = AnimatData.from_file(os.path.join(log_path, 'simulation.hdf5'))
        pylog.debug('Data successfully saved and logged back: {}'.format(data))

    # Plot network
    if prompt('Show connectivity maps', False):
        plot_networks_maps(animat_options.morphology, sim.animat().data)

    # Plot
    plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
