#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import farms_pylog as pylog
from farms_models.utils import get_sdf_path, get_simulation_data_path
from farms_data.amphibious.animat_data import AnimatData
from farms_bullet.utils.profile import profile
from farms_bullet.model.control import ControlType
from farms_bullet.simulation.options import SimulationOptions
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.utils.utils import prompt
from farms_amphibious.utils.network import plot_networks_maps
from farms_amphibious.experiment.simulation import simulation
from farms_amphibious.experiment.options import (
    amphibious_options,
    get_salamander_options,
)


def main(animat='salamander', version='v3', scale=0.2):
    """Main"""

    # Animat
    sdf = get_sdf_path(name=animat, version=version)
    pylog.info('Model SDF: {}'.format(sdf))
    animat_options = get_salamander_options(
        # n_joints_body=11,
        # spawn_position=[-5, 0, 0.1],
        # spawn_orientation=[0, 0, np.pi],
        # drives_init=[4.9, 0],
        # drives_init=[2.95, 0],
        drives_init=[2.0, 0],
        spawn_position=[-0.1*scale, 0, scale*0.05],
        spawn_orientation=[0, 0, 0],
        # drives_init=[4, 0],
        # spawn_position=[-0.3-0.5*scale, 0, scale*0.05],
        # default_control_type=ControlType.TORQUE,
        # muscle_alpha=3e-3,
        # muscle_beta=-1e-6,
        # muscle_gamma=5e3,
        # muscle_delta=-1e-8,
        # muscle_alpha=1e-3,
        # muscle_beta=-1e-6,
        # muscle_gamma=1e3,
        # muscle_delta=-1e-6,
        # weight_sens_contact_intralimb=0,
        # weight_sens_contact_opposite=0,
        # weight_sens_contact_following=0,
        # weight_sens_contact_diagonal=0,
        # weight_sens_hydro_freq=0,
        # weight_sens_hydro_amp=0,
        # modular_phases=np.array([3*np.pi/2, 0, 3*np.pi/2, 0]) - np.pi/4,
        # modular_amplitudes=np.full(4, 0.9),
    )

    for link in animat_options['morphology']['links']:
        link['pybullet_dynamics']['restitution'] = 0.0
        link['pybullet_dynamics']['lateralFriction'] = 1.0
        link['pybullet_dynamics']['spinningFriction'] = 0.0
        link['pybullet_dynamics']['rollingFriction'] = 0.0
    for joint_i, joint in enumerate(animat_options['morphology']['joints']):
        joint['pybullet_dynamics']['jointDamping'] = 0
        joint['pybullet_dynamics']['maxJointVelocity'] = 1e8  # 0.1
        joint['pybullet_dynamics']['jointLowerLimit'] = -1e8  # -0.1
        joint['pybullet_dynamics']['jointUpperLimit'] = +1e8  # +0.1
        joint['pybullet_dynamics']['jointLimitForce'] = 1e8
        joint_control = animat_options['control']['joints'][joint_i]
        assert joint['name'] == joint_control['joint']
        joint['initial_position'] = joint_control['bias']
        print('{}: {} [rad]'.format(joint['name'], joint_control['bias']))

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
    animat_options.show_hydrodynamics = not simulation_options.headless

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
        name=animat,
        version=version,
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