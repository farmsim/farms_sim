#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import pstats
import cProfile
import numpy as np
from farms_models.utils import get_sdf_path
from farms_bullet.simulation.options import SimulationOptions
from farms_bullet.model.model import (
    SimulationModels,
    DescriptionFormatModel
)
import farms_pylog as pylog
from farms_amphibious.model.animat import Amphibious
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.model.data import (
    AmphibiousOscillatorNetworkState,
    AmphibiousData
)
from farms_amphibious.simulation.simulation import AmphibiousSimulation
from farms_amphibious.network.network import AmphibiousNetworkODE


def get_animat_options(swimming=False):
    """Get animat options - Should load a config file in the future"""
    scale = 1
    animat_options = AmphibiousOptions(
        # collect_gps=True,
        show_hydrodynamics=True,
        scale=scale
    )
    # animat_options.control.drives.forward = 4

    if swimming:
        # Swiming
        animat_options.spawn.position = [-10, 0, 0]
        animat_options.spawn.orientation = [0, 0, np.pi]
    else:
        # Walking
        animat_options.spawn.position = [0, 0, scale*0.1]
        animat_options.spawn.orientation = [0, 0, 0]
        animat_options.physics.viscous = True
        animat_options.physics.buoyancy = True
        animat_options.physics.water_surface = True


    return animat_options


def get_simulation_options():
    """Get simulation options - Should load a config file in the future"""
    simulation_options = SimulationOptions.with_clargs()
    simulation_options.units.meters = 1
    simulation_options.units.seconds = 1e3
    simulation_options.units.kilograms = 1
    simulation_options.arena = "water"

    # Camera options
    simulation_options.video_yaw = 0
    simulation_options.video_pitch = -30
    simulation_options.video_distance = 1
    # simulation_options.video_name = (
    #     "transition_videos/swim2walk_y{}_p{}_d{}".format(
    #         simulation_options.video_yaw,
    #         simulation_options.video_pitch,
    #         simulation_options.video_distance,
    #     )
    # )

    return simulation_options


def get_flat_arena():
    """Flat arena"""
    return DescriptionFormatModel(
        path=get_sdf_path(
            name='arena_flat',
            version='v0',
        ),
        visual_options={
            'path': 'BIOROB2_blue.png',
            'rgbaColor': [1, 1, 1, 1],
            'specularColor': [1, 1, 1],
        }
    )


def get_water_arena(water_surface):
    """Water arena"""
    return SimulationModels([
        DescriptionFormatModel(
            path=get_sdf_path(
                name='arena_ramp',
                version='angle_-10_texture',
            ),
            visual_options={
                'path': 'BIOROB2_blue.png',
                'rgbaColor': [1, 1, 1, 1],
                'specularColor': [1, 1, 1],
            }
        ),
        DescriptionFormatModel(
            path=get_sdf_path(
                name='arena_water',
                version='v0',
            ),
            spawn_options={
                'posObj': [0, 0, water_surface],
                'ornObj': [0, 0, 0, 1],
            }
        ),
    ])


def simulation(sdf, **kwargs):
    """Siulation"""

    # Get options
    show_progress = True
    animat_options = kwargs.pop(
        'animat_options',
        get_animat_options(swimming=False)
    )
    simulation_options = kwargs.pop(
        'simulation_options',
        get_simulation_options()
    )

    # Creating arena
    if kwargs.pop('water_arena', False):
        animat_options.physics.water_surface = -0.1
        arena = get_water_arena(
            water_surface=animat_options.physics.water_surface
        )
    else:
        animat_options.physics.water_surface = None
        animat_options.physics.viscous = False
        animat_options.physics.sph = False
        animat_options.physics.resistive = False
        arena = get_flat_arena()

    # Animat options
    animat_options.morphology.n_dof_legs = 4
    animat_options.morphology.n_legs = 4

    # Animat sensors

    # Animat data
    animat_data = AmphibiousData.from_options(
        AmphibiousOscillatorNetworkState.default_state(
            simulation_options.n_iterations(),
            animat_options
        ),
        animat_options,
        simulation_options.n_iterations()
    )

    # Animat controller
    animat_controller = AmphibiousNetworkODE(
        animat_options=animat_options,
        animat_data=animat_data,
        timestep=simulation_options.timestep
    ) if kwargs.pop('use_controller', False) else None
    # AmphibiousKinematics(animat_options, animat_data, timestep),

    # Creating animat
    animat = Amphibious(
        sdf=sdf,
        options=animat_options,
        controller=animat_controller,
        timestep=simulation_options.timestep,
        iterations=simulation_options.n_iterations(),
        units=simulation_options.units,
    )

    # Setup simulation
    pylog.info("Creating simulation")
    sim = AmphibiousSimulation(
        simulation_options=simulation_options,
        animat=animat,
        arena=arena,
    )
    assert not kwargs

    # Run simulation
    pylog.info("Running simulation")
    # sim.run(show_progress=show_progress)
    # contacts = sim.models.animat.data.sensors.contacts
    for iteration in sim.iterator(show_progress=show_progress):
        # pylog.info(np.asarray(
        #     contacts.reaction(iteration, 0)
        # ))
        assert iteration >= 0

    # Analyse results
    pylog.info("Analysing simulation")
    sim.postprocess(
        iteration=sim.iteration,
        plot=simulation_options.plot,
        log_path=simulation_options.log_path,
        log_extension=simulation_options.log_extension,
        record=sim.options.record and not sim.options.headless
    )
    if simulation_options.log_path:
        np.save(
            simulation_options.log_path+"/hydrodynamics.npy",
            sim.models.animat.data.sensors.hydrodynamics.array
        )

    # Terminate simulation
    sim.end()


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
