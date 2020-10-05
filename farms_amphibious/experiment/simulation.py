#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import numpy as np
import farms_pylog as pylog
from farms_data.amphibious.data import AmphibiousData
from farms_bullet.control.kinematics import KinematicsController
from ..simulation.simulation import AmphibiousSimulation
from ..control.controller import AmphibiousController
from ..model.animat import Amphibious
from .options import get_animat_options, get_simulation_options


def simulation_setup(animat_sdf, arena, **kwargs):
    """Simulation setup"""
    # Get options
    animat_options = kwargs.pop(
        'animat_options',
        get_animat_options(swimming=False)
    )
    simulation_options = kwargs.pop(
        'simulation_options',
        get_simulation_options()
    )

    # Animat data
    animat_data = AmphibiousData.from_options(
        animat_options.control,
        simulation_options.n_iterations,
        simulation_options.timestep,
    )

    # Animat controller
    if kwargs.pop('use_controller', False):
        animat_controller = (
            KinematicsController(
                joints=animat_options.morphology.joints_names(),
                kinematics=np.loadtxt(animat_options.control.kinematics_file),
                sampling=animat_options.control.kinematics_sampling,
                timestep=simulation_options.timestep,
                n_iterations=simulation_options.n_iterations,
                animat_data=animat_data,
            )
            if animat_options.control.kinematics_file
            else AmphibiousController(
                joints=animat_options.morphology.joints_names(),
                animat_options=animat_options,
                animat_data=animat_data,
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


def simulation(animat_sdf, arena, show_progress=True, **kwargs):
    """Simulation"""

    # Instatiate simulation
    pylog.info('Creating simulation')
    sim = simulation_setup(animat_sdf, arena, **kwargs)

    # Run simulation
    pylog.info('Running simulation')
    # sim.run(show_progress=show_progress)
    # contacts = sim.models.animat.data.sensors.contacts
    for iteration in sim.iterator(show_progress=show_progress):
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
