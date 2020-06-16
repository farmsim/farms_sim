#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import os
import time
# import numpy as np
import matplotlib.pyplot as plt

import farms_pylog as pylog
from farms_models.utils import get_sdf_path, get_simulation_data_path
from farms_data.amphibious.animat_data import AnimatData
from farms_bullet.simulation.options import SimulationOptions
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.utils.utils import prompt
from farms_amphibious.utils.network import plot_networks_maps
from farms_amphibious.experiment.simulation import simulation, profile
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

    # State
    # state_init = animat_options.control.network.state_init
    # for phase_i, phase in enumerate(np.linspace(2*np.pi, 0, 11)):
    #     state_init[2*phase_i] = float(phase)
    #     state_init[2*phase_i+1] = float(phase)+np.pi
    # state_init = animat_options.control.network.state_init
    # for osc_i in range(4*animat_options.morphology.n_joints()):
    #     state_init[osc_i] = 1e-4*np.random.ranf()
    # n_joints = animat_options.morphology.n_joints()
    # state_init = (1e-4*np.random.ranf(5*n_joints)).tolist()
    # for osc_i, osc in enumerate(animat_options.control.network.oscillators):
    #     osc.initial_phase = state_init[osc_i]
    #     osc.initial_amplitude = state_init[osc_i+n_joints]

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
    log_path = get_simulation_data_path(
        name='salamander',
        version='v1',
        simulation_name='default',
    )
    video_name = os.path.join(log_path, 'simulation.mp4')
    save_data = prompt('Save data', False)
    if log_path and not os.path.isdir(log_path):
        os.mkdir(log_path)
    show_plots = prompt('Show plots', False)
    sim.postprocess(
        iteration=sim.iteration,
        log_path=log_path if save_data else '',
        plot=show_plots,
        video=video_name if sim.options.record else ''
    )
    if save_data:
        pylog.debug('Data saved, now loading back to check validity')
        data = AnimatData.from_file(os.path.join(log_path, 'simulation.hdf5'))
        pylog.debug('Data successfully saved and logged back: {}'.format(data))

    # Plot network
    show_connectivity = prompt('Show connectivity maps', False)
    if show_connectivity:
        plot_networks_maps(animat_options.morphology, sim.animat().data)

    # Plot
    if (show_plots or show_connectivity) and prompt('Save plots', False):
        extension = 'pdf'
        for fig in [plt.figure(num) for num in plt.get_fignums()]:
            filename = '{}.{}'.format(
                os.path.join(log_path, fig.canvas.get_window_title()),
                extension,
            )
            filename = filename.replace(' ', '_')
            pylog.debug('Saving to {}'.format(filename))
            fig.savefig(filename, format=extension)
    if show_plots or (
            show_connectivity
            and prompt('Show connectivity plots', False)
    ):
        plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
