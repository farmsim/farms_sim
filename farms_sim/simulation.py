#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

from farms_core import pylog
from farms_core.experiment.data import ExperimentData
from farms_core.model.options import AnimatOptions, ArenaOptions
from farms_core.simulation.options import Simulator, SimulationOptions
from farms_core.experiment.options import ExperimentOptions

from .utils.parse_args import sim_parse_args
from .utils.prompt import prompt_postprocessing

ENGINE_MUJOCO = False
try:
    from farms_mujoco.simulation.simulation import (
        Simulation as MuJoCoSimulation,
    )
    ENGINE_MUJOCO = True
except ModuleNotFoundError:
    # Package not installed - this is expected
    MuJoCoSimulation = None
except ImportError as e:
    # Package exists but has internal import issues - re-raise with context
    raise ImportError(f"farms_mujoco is installed but failed to import: {e}") from e

ENGINE_BULLET = False
try:
    from farms_bullet.simulation.simulation import (
        AnimatSimulation as PybulletSimulation
    )
    ENGINE_BULLET = True
except ModuleNotFoundError:
    # Package not installed - this is expected
    PybulletSimulation = None
except ImportError as e:
    # Package exists but has internal import issues - re-raise with context
    raise ImportError(f"farms_bullet is installed but failed to import: {e}") from e

if not ENGINE_MUJOCO and not ENGINE_BULLET:
    raise ModuleNotFoundError('Neither MuJoCo nor Bullet packages are installed')


def setup_from_clargs(clargs=None, **kwargs):
    """Simulation setup from clargs"""

    # Arguments
    if clargs is None:
        clargs = sim_parse_args()

    # Experiment options
    pylog.info('Getting experiment options')
    assert clargs.experiment_config, 'No experiment config provided'
    exp_loader = kwargs.pop('experiment_options_loader', ExperimentOptions)
    experiment_options = exp_loader.load(clargs.experiment_config)

    # Simulator
    simulator = {
        'MUJOCO': Simulator.MUJOCO,
        'PYBULLET': Simulator.PYBULLET,
    }[clargs.simulator]

    # Test options saving and loading
    if clargs.test_configs:
        sim_options = experiment_options.simulation
        animat_options = experiment_options.animats[0]
        # Save options
        animat_options_filename = 'animat_options.yaml'
        animat_options.save(animat_options_filename)
        sim_options_filename = 'simulation_options.yaml'
        sim_options.save(sim_options_filename)
        # Load options
        animat_options = animat_options_loader.load(animat_options_filename)
        sim_options = SimulationOptions.load(sim_options_filename)

    return clargs, experiment_options, simulator


def simulation_setup(
        experiment_options: ExperimentOptions,
        **kwargs,
) -> MuJoCoSimulation | PybulletSimulation:
    """Simulation setup"""

    # Get options
    simulator = kwargs.pop('simulator', Simulator.MUJOCO)
    handle_exceptions = kwargs.pop('handle_exceptions', False)

    # Experiment data
    experiment_data_class = kwargs.pop('experiment_data_class', ExperimentData)
    experiment_data = kwargs.pop(
        'experiment_data',
        experiment_data_class.from_options(experiment_options),
    )
    sim_options = experiment_options.simulation
    arena_options = experiment_options.arenas[0]

    # Simulator specific options
    if simulator == Simulator.MUJOCO:
        extensions = kwargs.pop('extensions', [])
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

        sim = MuJoCoSimulation.from_experiment(
            # Experiment
            experiment_options=experiment_options,
            # Models
            data=experiment_data,
            # Simulation
            restart=False,
            # Task
            extensions=extensions,
            handle_exceptions=handle_exceptions,
            # Save XML directly
            save_mjcf=save_mjcf,
            buffer_size=sim_options.runtime.buffer_size,
        )

    return sim


def run_simulation(
        experiment_options: ExperimentOptions,
        **kwargs,
) -> MuJoCoSimulation | PybulletSimulation:
    """Simulation"""

    # Instatiate simulation
    pylog.info('Creating simulation')
    simulator = kwargs.get('simulator', Simulator.MUJOCO)
    sim = simulation_setup(experiment_options, **kwargs)

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
        clargs = sim_parse_args()  # Parse command-line arguments
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
