"""Experiments options"""

import numpy as np
import farms_pylog as pylog
from farms_models.utils import get_sdf_path
from farms_bullet.simulation.options import SimulationOptions
from farms_bullet.model.model import SimulationModels, DescriptionFormatModel
from ..model.options import AmphibiousOptions


def get_animat_options(swimming=False, **kwargs):
    """Get animat options - Should load a config file in the future"""
    options = dict(
        spawn_position=[-10, 0, 0],
        spawn_orientation=[0, 0, 0],
        drag=True,
        buoyancy=True,
        water_surface=True,
        drives_init=[4, 0],
    ) if swimming else dict(
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, 0],
        drag=True,
        buoyancy=True,
        water_surface=True,
    )
    options.update(kwargs)
    return AmphibiousOptions.from_options(options)


def get_simulation_options(**kwargs):
    """Get simulation options - Should load a config file in the future"""
    simulation_options = SimulationOptions.with_clargs(**kwargs)
    simulation_options.units.meters = 1
    simulation_options.units.seconds = 1
    simulation_options.units.kilograms = 1

    # Camera options
    # simulation_options.video_yaw = 0
    # simulation_options.video_pitch = -30
    # simulation_options.video_distance = 1
    # simulation_options.video_name = (
    #     'transition_videos/swim2walk_y{}_p{}_d{}'.format(
    #         simulation_options.video_yaw,
    #         simulation_options.video_pitch,
    #         simulation_options.video_distance,
    #     )
    # )

    return simulation_options


def set_no_swimming_options(animat_options):
    """Set walking options"""
    animat_options.physics.water_surface = None
    animat_options.physics.drag = False


def set_swimming_options(animat_options, water_surface):
    """Set swimming options"""
    animat_options.physics.water_surface = water_surface


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


def amphibious_options(animat_options, use_water_arena=True, **kwargs):
    """Amphibious simulation"""

    # Arena
    if use_water_arena:
        arena = get_water_arena(water_surface=-0.1)
        set_swimming_options(animat_options, water_surface=-0.1)
    else:
        arena = get_flat_arena()
        set_no_swimming_options(animat_options)

    # Simulation
    simulation_options = get_simulation_options(**kwargs)

    return (simulation_options, arena)


def get_pleurobot_options(**kwargs):
    """Pleurobot default options"""

    # Animat
    sdf = get_sdf_path(name='pleurobot', version='0')
    pylog.info('Model SDF: {}'.format(sdf))

    # Morphology information
    links_names = kwargs.pop(
        'links_names',
        ['Head'] + [  # 'base_link',
            'link{}'.format(i+1)
            for i in range(6)
        ] + ['link_tailBone'] + [
            'link{}'.format(i+1)
            for i in range(6, 11)
        ] + ['link_tail'] + [
            'link{}'.format(i+1)
            for i in range(11, 27)
        ]
    )
    joints_names = kwargs.pop('joints_names', [
        # 'base_link_fixedjoint',
        'jHead',
        'j2',
        'j3',
        'j4',
        'j5',
        'j6',
        'j_tailBone',
        'j7',
        'j8',
        'j9',
        'j10',
        'j11',
        'j_tail',
        'ForearmLeft_Yaw',
        'ForearmLeft_Pitch',
        'ForearmLeft_Roll',
        'ForearmLeft_Elbow',
        'ForearmRight_Yaw',
        'ForearmRight_Pitch',
        'ForearmRight_Roll',
        'ForearmRight_Elbow',
        'HindLimbLeft_Yaw',
        'HindLimbLeft_Pitch',
        'HindLimbLeft_Roll',
        'HindLimbLeft_Elbow',
        'HindLimbRight_Yaw',
        'HindLimbRight_Pitch',
        'HindLimbRight_Roll',
        'HindLimbRight_Elbow',
    ])
    feet = kwargs.pop(
        'feet',
        ['link{}'.format(i+1) for i in [14, 18, 22, 26]]
    )
    links_no_collisions = kwargs.pop('links_no_collisions', [
        link
        for link in links_names
        if link not in feet+['Head', 'link_tail']
    ])

    # Joint options
    gain_amplitude = kwargs.pop('gain_amplitude', None)
    gain_offset = kwargs.pop('gain_offset', None)
    joints_offsets = kwargs.pop('joints_offsets', None)

    # Amplitudes gains
    if gain_amplitude is None:
        gain_amplitude = [-1]*(13+4*4)  # np.ones(13+4*4)
        gain_amplitude[6] = 0
        gain_amplitude[12] = 0
        for leg_i in range(2):
            for side_i in range(2):
                mirror = (-1 if side_i else 1)
                # mirror_full = (1 if leg_i else -1)*(1 if side_i else -1)
                gain_amplitude[13+2*leg_i*4+side_i*4+0] = mirror
                gain_amplitude[13+2*leg_i*4+side_i*4+1] = mirror
                gain_amplitude[13+2*leg_i*4+side_i*4+2] = -mirror
                gain_amplitude[13+2*leg_i*4+side_i*4+3] = mirror
        gain_amplitude = dict(zip(joints_names, gain_amplitude))

    # Offsets gains
    if gain_offset is None:
        gain_offset = [1]*(13+4*4)
        gain_offset[6] = 0
        gain_offset[12] = 0
        # for leg_i in range(2):
        #     for side_i in range(2):
        #         mirror = (-1 if side_i else 1)
        #         mirror_full = (1 if leg_i else -1)*(1 if side_i else -1)
        #         gain_offset[13+2*leg_i*4+side_i*4+0] = mirror
        #         gain_offset[13+2*leg_i*4+side_i*4+1] = mirror
        #         gain_offset[13+2*leg_i*4+side_i*4+2] = mirror_full
        #         gain_offset[13+2*leg_i*4+side_i*4+3] = mirror_full
        gain_offset = dict(zip(joints_names, gain_offset))

    # Joints joints_offsets
    if joints_offsets is None:
        joints_offsets = [0]*(13+4*4)
        for leg_i in range(2):
            for side_i in range(2):
                mirror = (1 if side_i else -1)
                mirror_full = (1 if leg_i else -1)*(1 if side_i else -1)
                joints_offsets[13+2*leg_i*4+side_i*4+0] = 0
                joints_offsets[13+2*leg_i*4+side_i*4+1] = mirror*np.pi/16 if leg_i else mirror*np.pi/8
                joints_offsets[13+2*leg_i*4+side_i*4+2] = 0 if leg_i else -mirror*np.pi/3
                joints_offsets[13+2*leg_i*4+side_i*4+3] = -mirror*np.pi/4 if leg_i else mirror*np.pi/16
        joints_offsets = dict(zip(joints_names, joints_offsets))

    # Animat options
    animat_options = get_animat_options(
        swimming=False,
        n_legs=4,
        n_dof_legs=4,
        n_joints_body=13,
        # body_head_amplitude=0,
        # body_tail_amplitude=0,
        body_stand_amplitude=kwargs.pop('body_stand_amplitude', 0.2),
        # body_stand_shift=np.pi/4,
        # legs_amplitude=[0.8, np.pi/32, np.pi/4, np.pi/8],
        # legs_offsets_walking=[0, np.pi/32, 0, np.pi/8],
        # legs_offsets_swimming=[-2*np.pi/5, 0, 0, 0],
        # body_stand_shift=kwargs.pop('body_stand_shift', np.pi/4),
        legs_amplitudes=kwargs.pop(
            'legs_amplitudes',
            [np.pi/8, np.pi/32, np.pi/8, np.pi/8],
        ),
        legs_offsets_walking=kwargs.pop(
            'legs_offsets_walking',
            [0, -np.pi/32, -np.pi/16, 0],
        ),
        legs_offsets_swimming=kwargs.pop(
            'legs_offsets_swimming',
            [2*np.pi/5, 0, 0, np.pi/2],
        ),
        gain_amplitude=gain_amplitude,
        gain_offset=gain_offset,
        offsets_bias=joints_offsets,
        weight_osc_body=kwargs.pop('weight_osc_body', 1e0),
        weight_osc_legs_internal=kwargs.pop('weight_osc_legs_internal', 3e1),
        weight_osc_legs_opposite=kwargs.pop('weight_osc_legs_opposite', 3e0),
        weight_osc_legs_following=kwargs.pop('weight_osc_legs_following', 3e0),
        weight_osc_legs2body=kwargs.pop('weight_osc_legs2body', 1e1),
        weight_sens_contact_intralimb=kwargs.pop('weight_sens_contact_intralimb', 0),
        weight_sens_contact_opposite=kwargs.pop('weight_sens_contact_opposite', 0),
        weight_sens_contact_following=kwargs.pop('weight_sens_contact_following', 0),
        weight_sens_contact_diagonal=kwargs.pop('weight_sens_contact_diagonal', 0),
        weight_sens_hydro_freq=kwargs.pop('weight_sens_hydro_freq', 0),
        weight_sens_hydro_amp=kwargs.pop('weight_sens_hydro_amp', 0),
        links_names=links_names,
        links_swimming=[],
        links_no_collisions=links_no_collisions,
        joints_names=joints_names,
        sensors_gps=links_names,
        sensors_joints=joints_names,
        sensors_contacts=feet,
        sensors_hydrodynamics=[],
        **kwargs
    )
    return sdf, animat_options


def fish_options(animat, version, kinematics_file, sampling_timestep, **kwargs):
    """Fish options"""
    pylog.info(kinematics_file)
    kinematics = np.loadtxt(kinematics_file)

    # Kinematics data handling
    n_sample = 50
    len_kinematics = np.shape(kinematics)[0]
    pose = kinematics[:, :3]
    position = np.ones(3)
    position[:2] = pose[0, :2]
    orientation = np.zeros(3)
    orientation[2] = pose[0, 2]
    velocity = np.zeros(3)
    velocity[:2] = pose[n_sample, :2] - pose[0, :2]
    velocity /= n_sample*sampling_timestep
    kinematics[:, 3:] = ((kinematics[:, 3:] + np.pi) % (2*np.pi)) - np.pi

    # Simulation options
    sim_options = {}
    if 'timestep' in kwargs:
        timestep = kwargs.pop('timestep')
        sim_options['timestep'] = timestep
    simulation_options = get_simulation_options(**sim_options)
    simulation_options.n_iterations = int(
        (len_kinematics-1)
        *sampling_timestep
        /simulation_options.timestep
    )

    # get_animat_options(swimming=False)
    simulation_options.gravity = [0, 0, 0]
    # simulation_options.timestep = 1e-3
    simulation_options.units.meters = 1
    simulation_options.units.seconds = 1
    simulation_options.units.kilograms = 1

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

    # Animat options
    n_joints = kinematics.shape[1]-3
    # links_names = ['link_body_0']+[
    #     'body_{}_t_link'.format(i+1)
    #     for i in range(n_joints-1)
    # ]
    links_names = ['link_body_0']+[
        '{}_v_{}_i_0_e_body_{}_t_link'.format(animat, version, i+1)
        for i in range(n_joints)
    ]
    # joints_names = ['joint_{}'.format(i) for i in range(n_joints)]
    joints_names = [
        '{}_v_{}_i_0_e_body_{}_t_link'.format(animat, version, i+1)
        for i in range(n_joints)
    ]
    animat_options = AmphibiousOptions.from_options(dict(
        show_hydrodynamics=True,
        n_legs=0,
        n_dof_legs=0,
        n_joints_body=n_joints,
        drag=kwargs.pop('drag', True),
        drag_coefficients=kwargs.pop('drag_coefficients', [
            np.array([-1e-5, -5e-2, -3e-2]),
            np.array([-1e-7, -1e-7, -1e-7]),
        ]),
        water_surface=kwargs.pop('water_surface', np.inf),
        links_names=links_names,
        links_swimming=links_names,
        links_no_collisions=links_names,
        joints_names=joints_names,
        sensors_gps=links_names,
        sensors_joints=joints_names,
        sensors_contacts=[],
        sensors_hydrodynamics=links_names,
        **kwargs
    ))

    # Arena
    arena = get_flat_arena()

    # Animat options
    animat_options.spawn.position = position.tolist()
    animat_options.spawn.orientation = orientation.tolist()
    animat_options.physics.buoyancy = False
    animat_options.spawn.velocity_lin = velocity.tolist()
    animat_options.spawn.velocity_ang = [0, 0, 0]
    animat_options.spawn.joints_positions = kinematics[0, 3:].tolist()
    animat_options.control.kinematics_file = kinematics_file
    # np.shape(kinematics)[1] - 3
    # animat_options.spawn.position = [-10, 0, 0]
    # animat_options.spawn.orientation = [0, 0, np.pi]

    return (
        animat_options,
        arena,
        simulation_options,
        kinematics,
    )
