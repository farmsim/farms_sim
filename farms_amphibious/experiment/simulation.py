#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import pstats
import cProfile
import numpy as np
import farms_pylog as pylog
from ..simulation.simulation import AmphibiousSimulation
from ..control.controller import AmphibiousController
from ..control.kinematics import AmphibiousKinematics
from ..model.animat import Amphibious
from ..model.data import AmphibiousOscillatorNetworkState, AmphibiousData
from .options import get_animat_options, get_simulation_options


def simulation_setup(animat_sdf, arena_sdf, **kwargs):
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

    # Animat sensors

    # Animat data
    animat_data = AmphibiousData.from_options(
        AmphibiousOscillatorNetworkState.default_state(
            simulation_options.n_iterations,
            animat_options.morphology,
        ),
        animat_options.morphology,
        animat_options.control,
        simulation_options.n_iterations
    )

    # Animat controller
    if kwargs.pop('use_controller', False):
        if animat_options.control.kinematics_file:
            animat_controller = AmphibiousKinematics(
                joints=animat_options.morphology.joints,
                animat_options=animat_options,
                animat_data=animat_data,
                timestep=simulation_options.timestep,
                n_iterations=simulation_options.n_iterations,
                sampling=kwargs.pop('sampling')
            )
        else:
            animat_controller = AmphibiousController(
                joints=animat_options.morphology.joints,
                animat_options=animat_options,
                animat_data=animat_data,
                timestep=simulation_options.timestep
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
        arena=arena_sdf,
    )
    return sim


def simulation_post(sim):
    """Simulation post-processing"""
    # Analyse results
    pylog.info('Analysing simulation')
    sim.postprocess(
        iteration=sim.iteration,
        plot=sim.options.plot,
        log_path=sim.options.log_path,
        log_extension=sim.options.log_extension,
        record=sim.options.record and not sim.options.headless
    )
    if sim.options.log_path:
        np.save(
            sim.options.log_path+'/hydrodynamics.npy',
            sim.animat().data.sensors.hydrodynamics.array
        )


def simulation(animat_sdf, arena_sdf, show_progress=True, **kwargs):
    """Simulation"""

    # Instatiate simulation
    pylog.info('Creating simulation')
    sim = simulation_setup(animat_sdf, arena_sdf, **kwargs)

    # Run simulation
    pylog.info('Running simulation')
    # sim.run(show_progress=show_progress)
    # contacts = sim.models.animat.data.sensors.contacts
    for iteration in sim.iterator(show_progress=show_progress):
        # pylog.info(np.asarray(
        #     contacts.reaction(iteration, 0)
        # ))
        assert iteration >= 0

    # Post-processing
    pylog.info('Simulation post-processing')
    simulation_post(sim)

    # Terminate simulation
    pylog.info('Terminating simulation')
    sim.end()
    return sim


def profile(function, **kwargs):
    """Profile with cProfile"""
    n_time = kwargs.pop('pstat_n_time', 30)
    n_cumtime = kwargs.pop('pstat_n_cumtime', 30)
    cProfile.runctx(
        statement='function(**kwargs)',
        globals={},
        locals={'function': function, 'kwargs': kwargs},
        filename='simulation.profile'
    )
    pstat = pstats.Stats('simulation.profile')
    pstat.sort_stats('time').print_stats(n_time)
    pstat.sort_stats('cumtime').print_stats(n_cumtime)
