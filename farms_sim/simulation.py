#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

from typing import Union
from farms_core import pylog
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions, ArenaOptions
from farms_core.simulation.options import Simulator, SimulationOptions

from .utils.parse_args import sim_parse_args
from .utils.prompt import prompt_postprocessing

ENGINE_MUJOCO = False
try:
    from farms_mujoco.simulation.simulation import (
        Simulation as MuJoCoSimulation,
    )
    ENGINE_MUJOCO = True
except ImportError:
    MuJoCoSimulation = None

ENGINE_BULLET = False
try:
    from farms_bullet.simulation.simulation import (
        AnimatSimulation as PybulletSimulation
    )
    ENGINE_BULLET = True
except ImportError:
    PybulletSimulation = None

if not ENGINE_MUJOCO and not ENGINE_BULLET:
    raise ImportError('Neither MuJoCo nor Bullet are installed')


def setup_from_clargs(clargs=None, **kwargs):
    """Simulation setup from clargs"""

    # Arguments
    if clargs is None:
        clargs = sim_parse_args()

    # Animat options
    pylog.info('Getting animat options')
    assert clargs.animat_config, 'No animat config provided'
    animat_options_loader = kwargs.pop('animat_options_loader', AnimatOptions)
    animat_options = animat_options_loader.load(clargs.animat_config)

    # Simulation options
    pylog.info('Getting simulation options')
    assert clargs.simulation_config, 'No simulation config provided'
    sim_options = SimulationOptions.load(clargs.simulation_config)

    # Arena options
    pylog.info('Getting arena options')
    assert clargs.arena_config, 'No arena config provided'
    arena_options_loader = kwargs.pop('arena_options_loader', ArenaOptions)
    arena_options = arena_options_loader.load(clargs.arena_config)

    # Simulator
    simulator = {
        'MUJOCO': Simulator.MUJOCO,
        'PYBULLET': Simulator.PYBULLET,
    }[clargs.simulator]

    # Test options saving and loading
    if clargs.test_configs:
        # Save options
        animat_options_filename = 'animat_options.yaml'
        animat_options.save(animat_options_filename)
        sim_options_filename = 'simulation_options.yaml'
        sim_options.save(sim_options_filename)
        # Load options
        animat_options = animat_options_loader.load(animat_options_filename)
        sim_options = SimulationOptions.load(sim_options_filename)

    return clargs, animat_options, sim_options, arena_options, simulator


def simulation_setup(
        animat_options: AnimatOptions,
        arena_options: ArenaOptions,
        **kwargs,
) -> Union[MuJoCoSimulation, PybulletSimulation]:
    """Simulation setup"""

    # Get options
    simulator = kwargs.pop('simulator', Simulator.MUJOCO)
    handle_exceptions = kwargs.pop('handle_exceptions', False)
    sim_options = kwargs.pop(
        'simulation_options',
        SimulationOptions.with_clargs(),
    )

    # Animat data
    animat_data_class = kwargs.pop('animat_data_class', AnimatData)
    animat_data = kwargs.pop(
        'animat_data',
        animat_data_class.from_options(
            animat_options=animat_options,
            simulation_options=sim_options,
        ),
    )

    # Animat controller
    animat_controller = kwargs.pop('animat_controller', None)

    # Simulator specific options
    if simulator == Simulator.MUJOCO:
        callbacks = kwargs.pop('callbacks', [])
        save_mjcf = kwargs.pop('save_mjcf', False)
    elif simulator == Simulator.PYBULLET:
        animat = kwargs.pop('animat', None)
        sim_loader = kwargs.pop('sim_loader', PybulletSimulation)

    # Kwargs check
    assert not kwargs, kwargs

    # Pybullet
    if simulator == Simulator.PYBULLET:

        # Setup simulation
        pylog.info('Creating simulation')
        sim = sim_loader(
            simulation_options=sim_options,
            animat=animat,
            arena_options=arena_options,
        )

    # Mujoco
    elif simulator == Simulator.MUJOCO:

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
            handle_exceptions=handle_exceptions,
            # Save XML directly
            save_mjcf=save_mjcf,
        )

    return sim


def run_simulation(
        animat_options: AnimatOptions,
        arena_options: ArenaOptions,
        **kwargs,
) -> Union[MuJoCoSimulation, PybulletSimulation]:
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


def postprocessing_from_clargs(sim, clargs=None, **kwargs):
    """Simulation postproces"""
    if clargs is None:
        clargs = sim_parse_args()
        kwargs['simulator'] = {
            'MUJOCO': Simulator.MUJOCO,
            'PYBULLET': Simulator.PYBULLET,
        }[clargs.simulator]
    prompt_postprocessing(
        sim=sim,
        query=clargs.prompt,
        log_path=clargs.log_path,
        verify=clargs.verify_save,
        **kwargs,
    )
