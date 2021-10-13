#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import numpy as np
import farms_pylog as pylog

from farms_data.amphibious.data import AmphibiousData
from farms_models.utils import get_sdf_path
from farms_bullet.model.options import SpawnLoader
from farms_bullet.model.control import ControlType
from farms_bullet.simulation.options import SimulationOptions
from farms_bullet.control.kinematics import KinematicsController

from ..model.animat import Amphibious
from ..model.options import AmphibiousOptions
from ..simulation.simulation import AmphibiousSimulation
from ..utils.parse_args import parse_args
from ..utils.prompt import prompt_postprocessing
from ..control.controller import AmphibiousController
from ..control.manta_control import MantaController
from ..control.drive import drive_from_config
from .options import (
    amphibious_options,
    get_animat_options_from_model,
)


def setup_from_clargs(clargs=None):
    """Simulation setup from clargs"""

    # Arguments
    if clargs is None:
        clargs = parse_args()

    # Options
    sdf = (
        clargs.sdf
        if clargs.sdf
        else get_sdf_path(name=clargs.animat, version=clargs.version)
    )
    kwargs = {}
    if clargs.torque_equation is not None:
        kwargs['torque_equation'] = clargs.torque_equation
    if clargs.weight_osc_body is not None:
        kwargs['weight_osc_body'] = clargs.weight_osc_body
    if clargs.weight_osc_body2legs is not None:
        kwargs['weight_osc_body2legs'] = clargs.weight_osc_body2legs
    if clargs.weight_osc_legs2body is not None:
        kwargs['weight_osc_legs2body'] = clargs.weight_osc_legs2body
    if clargs.weight_sens_stretch_freq is not None:
        kwargs['weight_sens_stretch_freq'] = clargs.weight_sens_stretch_freq
    animat_options = get_animat_options_from_model(
        animat=clargs.animat,
        version=clargs.version,
        default_max_torque=clargs.max_torque,
        # max_torque=clargs.max_torque,
        max_velocity=clargs.max_velocity,
        default_lateral_friction=clargs.lateral_friction,
        feet_friction=clargs.feet_friction,
        default_restitution=clargs.default_restitution,
        use_self_collisions=clargs.self_collisions,
        spawn_loader={
            'FARMS': SpawnLoader.FARMS,
            'PYBULLET': SpawnLoader.PYBULLET,
        }[clargs.spawn_loader],
        drives_init=clargs.drives,
        spawn_position=clargs.position,
        spawn_orientation=clargs.orientation,
        default_equation=clargs.control_type,
        **kwargs,
    )
    simulation_options, arena = amphibious_options(
        animat_options=animat_options,
        arena=clargs.arena,
        arena_sdf=clargs.arena_sdf,
        water_sdf=clargs.water_sdf,
        water_height=clargs.water_height,
        water_velocity=clargs.water_velocity,
        viscosity=clargs.viscosity,
        ground_height=clargs.ground_height,
    )

    # Test options saving and loading
    if clargs.test_configs:
        # Save options
        animat_options_filename = 'animat_options.yaml'
        animat_options.save(animat_options_filename)
        simulation_options_filename = 'simulation_options.yaml'
        simulation_options.save(simulation_options_filename)
        # Load options
        animat_options = AmphibiousOptions.load(animat_options_filename)
        simulation_options = SimulationOptions.load(simulation_options_filename)

    return clargs, sdf, animat_options, simulation_options, arena


def simulation_setup(animat_sdf, animat_options, arena, **kwargs):
    """Simulation setup"""
    # Get options
    simulation_options = kwargs.pop(
        'simulation_options',
        SimulationOptions.with_clargs()
    )

    # Animat data
    animat_data = kwargs.pop(
        'animat_data',
        AmphibiousData.from_options(
            animat_options.control,
            simulation_options.n_iterations,
            simulation_options.timestep,
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
                timestep=simulation_options.timestep,
                n_iterations=simulation_options.n_iterations,
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
                        simulation_options=simulation_options,
                    )
                    if drive_config
                    else None
                ),
            )
        )
    else:
        animat_controller = None

    # Creating animat
    animat = Amphibious(
        sdf=animat_sdf,
        options=animat_options,
        controller=animat_controller,
        timestep=simulation_options.timestep,
        iterations=simulation_options.n_iterations,
        units=simulation_options.units,
    )

    # Setup simulation
    assert not kwargs, 'Unknown kwargs:\n{}'.format(kwargs)
    pylog.info('Creating simulation')
    sim = AmphibiousSimulation(
        simulation_options=simulation_options,
        animat=animat,
        arena=arena,
    )
    return sim


def simulation(animat_sdf, animat_options, arena, **kwargs):
    """Simulation"""

    # Instatiate simulation
    pylog.info('Creating simulation')
    sim = simulation_setup(animat_sdf, animat_options, arena, **kwargs)

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

    return sim


def simulation_post(sim, log_path='', plot=False, video=''):
    """Simulation post-processing"""
    sim.postprocess(
        iteration=sim.iteration,
        log_path=log_path,
        plot=plot,
        video=video if not sim.options.headless else ''
    )


def postprocessing_from_clargs(sim, animat_options, clargs=None):
    """Simulation postproces"""
    if clargs is None:
        clargs = parse_args()
    prompt_postprocessing(
        animat=clargs.animat,
        version=clargs.version,
        sim=sim,
        animat_options=animat_options,
        query=clargs.prompt,
        save=clargs.save,
        save_to_models=clargs.save_to_models,
        verify=clargs.verify_save,
    )
