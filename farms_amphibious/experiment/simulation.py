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
from farms_amphibious.model.convention import AmphibiousConvention
from farms_amphibious.model.data import (
    AmphibiousOscillatorNetworkState,
    AmphibiousData
)
from farms_amphibious.simulation.simulation import AmphibiousSimulation
from farms_amphibious.network.network import AmphibiousNetworkODE
from farms_amphibious.network.kinematics import AmphibiousKinematics

def get_animat_options(swimming=False, **kwargs):
    """Get animat options - Should load a config file in the future"""
    scale = 1
    animat_options = AmphibiousOptions(
        # collect_gps=True,
        show_hydrodynamics=True,
        **kwargs
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


def get_simulation_options(**kwargs):
    """Get simulation options - Should load a config file in the future"""
    simulation_options = SimulationOptions.with_clargs(**kwargs)
    simulation_options.units.meters = 1
    simulation_options.units.seconds = 1e3
    simulation_options.units.kilograms = 1
    simulation_options.arena = 'water'

    # Camera options
    simulation_options.video_yaw = 0
    simulation_options.video_pitch = -30
    simulation_options.video_distance = 1
    # simulation_options.video_name = (
    #     'transition_videos/swim2walk_y{}_p{}_d{}'.format(
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


def set_no_swimming_options(animat_options):
    """Set walking options"""
    animat_options.physics.water_surface = None
    animat_options.physics.viscous = False
    animat_options.physics.sph = False
    animat_options.physics.resistive = False


def set_swimming_options(animat_options, water_surface):
    """Set swimming options"""
    animat_options.physics.water_surface = water_surface


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
            simulation_options.n_iterations(),
            animat_options
        ),
        animat_options,
        simulation_options.n_iterations()
    )

    # Animat controller
    if kwargs.pop('use_controller', False):
        if animat_options.control.kinematics_file:
            animat_controller = AmphibiousKinematics(
                animat_options=animat_options,
                animat_data=animat_data,
                timestep=simulation_options.timestep,
                n_iterations=simulation_options.n_iterations(),
                sampling=kwargs.pop('sampling')
            )
        else:
            animat_controller = AmphibiousNetworkODE(
                animat_options=animat_options,
                animat_data=animat_data,
                timestep=simulation_options.timestep
            )
    else:
        animat_controller = None

    # Creating animat
    feet = kwargs.pop('feet', None)
    links = kwargs.pop('links', None)
    joints = kwargs.pop('joints', None)
    links_swimming = kwargs.pop('links_swimming', None)
    links_no_collisions = kwargs.pop('links_no_collisions', None)
    animat = Amphibious(
        sdf=animat_sdf,
        options=animat_options,
        controller=animat_controller,
        timestep=simulation_options.timestep,
        iterations=simulation_options.n_iterations(),
        units=simulation_options.units,
        links=links,
        joints=joints,
        feet=feet,
        links_swimming=links_swimming,
        links_no_collisions=links_no_collisions,
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

    pylog.info('Simulation post-processing')
    simulation_post(sim)

    # Terminate simulation
    pylog.info('Terminating simulation')
    sim.end()
    return sim


def amphibious_options(animat_options, use_water_arena=True):
    """Amphibious simulation"""

    # Arena
    if use_water_arena:
        arena_sdf = get_water_arena(water_surface=-0.1)
        set_swimming_options(animat_options, water_surface=-0.1)
    else:
        arena_sdf = get_flat_arena()
        set_no_swimming_options(animat_options)

    # Simulation
    simulation_options = get_simulation_options()

    convention = AmphibiousConvention(animat_options)
    links = [
        convention.bodylink2name(i)
        for i in range(animat_options.morphology.n_links_body())
    ] + [
        convention.leglink2name(leg_i, side_i, link_i)
        for leg_i in range(animat_options.morphology.n_legs//2)
        for side_i in range(2)
        for link_i in range(animat_options.morphology.n_dof_legs)
    ]
    joints = [
        convention.bodyjoint2name(i)
        for i in range(animat_options.morphology.n_joints_body)
    ] + [
        convention.legjoint2name(leg_i, side_i, joint_i)
        for leg_i in range(animat_options.morphology.n_legs//2)
        for side_i in range(2)
        for joint_i in range(animat_options.morphology.n_dof_legs)
    ]
    feet = [
        convention.leglink2name(
            leg_i=leg_i,
            side_i=side_i,
            joint_i=animat_options.morphology.n_dof_legs-1
        )
        for leg_i in range(animat_options.morphology.n_legs//2)
        for side_i in range(2)
    ]
    links_swimming = [
        convention.bodylink2name(body_i)
        for body_i in range(animat_options.morphology.n_links_body())
    ]
    links_no_collisions = [
        convention.bodylink2name(body_i)
        for body_i in range(0)
    ] + [
        convention.leglink2name(leg_i, side_i, joint_i)
        for leg_i in range(animat_options.morphology.n_legs//2)
        for side_i in range(2)
        for joint_i in range(animat_options.morphology.n_dof_legs-1)
    ]
    return (
        simulation_options,
        arena_sdf,
        links,
        joints,
        feet,
        links_swimming,
        links_no_collisions,
    )


def fish_options(kinematics_file, sampling_timestep, timestep=1e-3, **kwargs):
    """Fish options"""
    pylog.info(kinematics_file)
    kinematics = np.loadtxt(kinematics_file)

    n_joints = kinematics.shape[1]-3
    animat_options = AmphibiousOptions(
        show_hydrodynamics=True,
        n_legs=0,
        n_dof_legs=0,
        n_joints_body=n_joints,
        viscous=kwargs.pop('viscous', False),
        resistive=kwargs.pop('resistive', True),
        resistive_coefficients=kwargs.pop('resistive_coefficients', [
            1e-1*np.array([-1e-4, -5e-1, -3e-1]),
            1e-1*np.array([-1e-6, -1e-6, -1e-6]),
        ]),
        water_surface=kwargs.pop('water_surface', np.inf),
        **kwargs
    )

    # Arena
    arena_sdf = get_flat_arena()

    # get_animat_options(swimming=False)
    simulation_options = get_simulation_options(timestep=timestep)
    simulation_options.gravity = [0, 0, 0]
    # simulation_options.timestep = 1e-3
    simulation_options.units.meters = 1
    simulation_options.units.seconds = 1e3
    simulation_options.units.kilograms = 1

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

    # Kinematics data handling
    n_sample = 100
    animat_options.control.kinematics_file = kinematics_file
    len_kinematics = np.shape(kinematics)[0]
    simulation_options.duration = (len_kinematics-1)*sampling_timestep
    pose = kinematics[:, :3]
    position = np.ones(3)
    position[:2] = pose[0, :2]
    orientation = np.zeros(3)
    orientation[2] = pose[0, 2]
    velocity = np.zeros(3)
    velocity[:2] = pose[n_sample, :2] - pose[0, :2]
    velocity /= n_sample*sampling_timestep
    # kinematics = kinematics[:, 3:]
    kinematics[:, 3:] = ((kinematics[:, 3:] + np.pi) % (2*np.pi)) - np.pi

    # Animat options
    animat_options.spawn.position = position
    animat_options.spawn.orientation = orientation
    animat_options.physics.buoyancy = False
    animat_options.spawn.velocity_lin = velocity
    animat_options.spawn.velocity_ang = [0, 0, 0]
    animat_options.spawn.joints_positions = kinematics[0, 3:]
    # np.shape(kinematics)[1] - 3
    # animat_options.spawn.position = [-10, 0, 0]
    # animat_options.spawn.orientation = [0, 0, np.pi]

    # Links
    n_joints = animat_options.morphology.n_joints()
    links = ['link_body_0']+[
        'body_{}_t_link'.format(i+1)
        for i in range(n_joints-1)
    ]

    # Joints
    joints = ['joint_{}'.format(i) for i in range(n_joints)]

    return (
        animat_options,
        arena_sdf,
        simulation_options,
        kinematics,
        links,
        joints
    )


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
