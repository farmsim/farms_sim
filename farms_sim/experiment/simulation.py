#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import numpy as np

import farms_pylog as pylog
from farms_data.model.options import ArenaOptions
from farms_data.amphibious.data import AmphibiousData
from farms_data.simulation.options import Simulator, SimulationOptions

from ..model.options import AmphibiousOptions
from ..utils.parse_args import sim_parse_args
from ..utils.prompt import prompt_postprocessing
from ..control.amphibious import AmphibiousController
from ..control.kinematics import KinematicsController
from ..control.manta_control import MantaController
from ..control.drive import drive_from_config

from .callbacks import SwimmingCallback

ENGINE_MUJOCO = False
try:
    from farms_mujoco.simulation.simulation import (
        Simulation as MuJoCoSimulation,
    )
    ENGINE_MUJOCO = True
except ImportError as err:
    pylog.error(err)
    ENGINE_MUJOCO = False
ENGINE_BULLET = False
try:
    from ..bullet.animat import Amphibious
    from ..bullet.simulation import AmphibiousPybulletSimulation
    ENGINE_BULLET = True
except ImportError as err:
    pylog.error(err)
    ENGINE_BULLET = False

if not ENGINE_MUJOCO and not ENGINE_BULLET:
    raise ImportError('Neither MuJoCo nor Bullet are installed')


def setup_from_clargs(clargs=None):
    """Simulation setup from clargs"""

    # Arguments
    if clargs is None:
        clargs = sim_parse_args()

    # Animat options
    pylog.info('Getting animat options')
    assert clargs.animat_config, 'No animat config provided'
    animat_options = AmphibiousOptions.load(clargs.animat_config)

    # Simulation options
    pylog.info('Getting simulation options')
    assert clargs.simulation_config, 'No simulation config provided'
    sim_options = SimulationOptions.load(clargs.simulation_config)

    # Arena options
    pylog.info('Getting arena options')
    assert clargs.arena_config, 'No arena config provided'
    arena_options = ArenaOptions.load(clargs.arena_config)

    # Test options saving and loading
    if clargs.test_configs:
        # Save options
        animat_options_filename = 'animat_options.yaml'
        animat_options.save(animat_options_filename)
        sim_options_filename = 'simulation_options.yaml'
        sim_options.save(sim_options_filename)
        # Load options
        animat_options = AmphibiousOptions.load(animat_options_filename)
        sim_options = SimulationOptions.load(sim_options_filename)

    return clargs, animat_options, sim_options, arena_options


def simulation_setup(
        animat_options: AmphibiousOptions,
        arena_options: ArenaOptions,
        **kwargs,
):
    """Simulation setup"""
    # Get options
    simulator = kwargs.pop('simulator', Simulator.MUJOCO)
    sim_options = kwargs.pop(
        'simulation_options',
        SimulationOptions.with_clargs()
    )

    # Animat data
    animat_data = kwargs.pop(
        'animat_data',
        AmphibiousData.from_options(
            control=animat_options.control,
            initial_state=animat_options.state_init(),
            n_iterations=sim_options.n_iterations,
            timestep=sim_options.timestep,
        ),
    )

    # Animat controller
    if kwargs.pop('use_controller', False) or 'animat_controller' in kwargs:
        drive_config = kwargs.pop('drive_config', None)
        animat_controller = (
            kwargs.pop('animat_controller')
            if 'animat_controller' in kwargs
            else KinematicsController(
                joints_names=animat_options.morphology.joints_names(),
                kinematics=np.genfromtxt(animat_options.control.kinematics_file),
                sampling=animat_options.control.kinematics_sampling,
                timestep=sim_options.timestep,
                n_iterations=sim_options.n_iterations,
                animat_data=animat_data,
                max_torques={
                    joint.joint: joint.max_torque
                    for joint in animat_options.control.joints
                },
            )
            if animat_options.control.kinematics_file
            else MantaController(
                joints_names=animat_options.morphology.joints_names(),
                animat_options=animat_options,
                animat_data=animat_data,
            )
            if animat_options.control.manta_controller
            else AmphibiousController(
                joints_names=animat_options.control.joints_names(),
                animat_options=animat_options,
                animat_data=animat_data,
                drive=(
                    drive_from_config(
                        filename=drive_config,
                        animat_data=animat_data,
                        simulation_options=sim_options,
                    )
                    if drive_config
                    else None
                ),
            )
        )
    else:
        animat_controller = None

    # Kwargs check
    assert not kwargs, kwargs

    # Pybullet
    if simulator == Simulator.PYBULLET:

        # Creating animat
        animat = Amphibious(
            options=animat_options,
            controller=animat_controller,
            timestep=sim_options.timestep,
            iterations=sim_options.n_iterations,
            units=sim_options.units,
        )

        # Setup simulation
        pylog.info('Creating simulation')
        sim = AmphibiousPybulletSimulation(
            simulation_options=sim_options,
            animat=animat,
            arena_options=arena_options,
        )

    # Mujoco
    elif simulator == Simulator.MUJOCO:

        # Callbacks
        callbacks = []

        # Hydrodynamics
        if animat_options.physics.drag or animat_options.physics.sph:
            callbacks += [SwimmingCallback(animat_options=animat_options)]

        sim = MuJoCoSimulation.from_sdf(
            # Models
            animat_options=animat_options,
            data=animat_data,
            controller=animat_controller,
            arena_options=arena_options,
            # Simulation
            simulation_options=sim_options,
            restart=False,
            # Task
            callbacks=callbacks,
        )

    return sim


def simulation(
        animat_options: AmphibiousOptions,
        arena_options: ArenaOptions,
        **kwargs,
):
    """Simulation"""

    # Instatiate simulation
    pylog.info('Creating simulation')
    simulator = kwargs.get('simulator', Simulator.MUJOCO)
    sim = simulation_setup(animat_options, arena_options, **kwargs)

    if simulator == Simulator.PYBULLET:

        # Run simulation
        pylog.info('Running simulation')
        # sim.run(show_progress=show_progress)
        # contacts = sim.models.animat.data.sensors.contacts
        for iteration in sim.iterator(show_progress=sim.options.show_progress):
            # pylog.info(np.asarray(
            #     contacts.reaction(iteration, 0)
            # ))
            assert iteration >= 0

        # Terminate simulation
        pylog.info('Terminating simulation')
        sim.end()

    elif simulator == Simulator.MUJOCO:

        # Run simulation
        pylog.info('Running simulation')
        sim.run()

    return sim


def simulation_post(sim, log_path='', plot=False, video=''):
    """Simulation post-processing"""
    sim.postprocess(
        iteration=sim.iteration,
        log_path=log_path,
        plot=plot,
        video=video if not sim.options.headless else ''
    )


def postprocessing_from_clargs(sim, animat_options, simulator, clargs=None):
    """Simulation postproces"""
    if clargs is None:
        clargs = sim_parse_args()
        simulator = {
            'MUJOCO': Simulator.MUJOCO,
            'PYBULLET': Simulator.PYBULLET,
        }[clargs.simulator]
    prompt_postprocessing(
        sim=sim,
        animat_options=animat_options,
        query=clargs.prompt,
        log_path=clargs.log_path,
        verify=clargs.verify_save,
        simulator=simulator,
    )
