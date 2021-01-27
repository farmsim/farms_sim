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
    get_salamander_options,
)


def main(animat='salamander', version='v3'):
    """Main"""

    # Arguments
    clargs = parse_args()

    # Options
    sdf = get_sdf_path(name=animat, version=version)
    pylog.info('Model SDF: {}'.format(sdf))
    animat_options = get_salamander_options(
        # n_joints_body=11,
        # spawn_position=[-5, 0, 0.1],
        # spawn_orientation=[0, 0, np.pi],
        # drives_init=[4.9, 0],
        # drives_init=[2.95, 0],
        # drives_init=[2.0, 0],
        # spawn_position=[-0.1*scale, 0, scale*0.05],
        # spawn_orientation=[0, 0, 0],
        # drives_init=[4, 0],
        # spawn_position=[-0.3-0.5*scale, 0, scale*0.05],
        spawn_loader=SpawnLoader.PYBULLET,
        # spawn_loader=SpawnLoader.FARMS,
        default_control_type=ControlType.POSITION,
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

    # for link in animat_options['morphology']['links']:
    #     link['pybullet_dynamics']['restitution'] = 0.0
    #     link['pybullet_dynamics']['lateralFriction'] = 0.1
    #     link['pybullet_dynamics']['spinningFriction'] = 0.0
    #     link['pybullet_dynamics']['rollingFriction'] = 0.0
    # for joint_i, joint in enumerate(animat_options['morphology']['joints']):
    #     joint['pybullet_dynamics']['jointDamping'] = 0
    #     joint['pybullet_dynamics']['maxJointVelocity'] = +np.inf  # 0.1
    #     joint['pybullet_dynamics']['jointLowerLimit'] = -np.inf  # -0.1
    #     joint['pybullet_dynamics']['jointUpperLimit'] = +np.inf  # +0.1
    #     joint['pybullet_dynamics']['jointLimitForce'] = +np.inf
    #     joint_control = animat_options['control']['joints'][joint_i]
    #     assert joint['name'] == joint_control['joint']
    #     joint['initial_position'] = joint_control['bias']
    #     print('{}: {} [rad]'.format(joint['name'], joint_control['bias']))

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
        use_water_arena=False,
    )
    simulation_options.units.meters = 1
    simulation_options.units.seconds = 1
    simulation_options.units.kilograms = 1
    animat_options.show_hydrodynamics = not simulation_options.headless

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
        animat=animat,
        version=version,
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
