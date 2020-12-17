"""Experiments options"""

import numpy as np
import farms_pylog as pylog
from farms_models.utils import get_sdf_path
from farms_bullet.simulation.options import SimulationOptions
from farms_bullet.model.model import SimulationModels, DescriptionFormatModel
from farms_bullet.model.options import SpawnLoader
from farms_bullet.model.control import ControlType
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
    animat_options.physics.sph = False
    animat_options.physics.drag = False
    animat_options.physics.water_surface = None


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

    # Water
    water_surface = kwargs.pop('water_surface', -0.1*0.1)

    # Arena
    if use_water_arena:
        arena = get_water_arena(water_surface=water_surface)
        set_swimming_options(animat_options, water_surface=water_surface)
    else:
        arena = get_flat_arena()
        set_no_swimming_options(animat_options)

    # Simulation
    simulation_options = get_simulation_options(**kwargs)

    return (simulation_options, arena)


def get_salamander_kwargs_options(**kwargs):
    """Salamander options"""
    n_joints_body = kwargs.pop('n_joints_body', 11)
    kwargs_options = {
        'spawn_loader': SpawnLoader.FARMS,  # SpawnLoader.PYBULLET,
        'spawn_position': [0, 0, 0.2*0.07],
        'spawn_orientation': [0, 0, 0],
        'use_self_collisions': True,
        'default_control_type': ControlType.POSITION,  # ControlType.TORQUE,
        'show_hydrodynamics': True,
        'swimming': False,
        'n_legs': 4,
        'n_dof_legs': 4,
        'n_joints_body': n_joints_body,
        'drag_coefficients': [
            [
                [-1e-1, -1e0, -1e0]
                if i < 12
                else [-1e-4, -1e-4, -1e-4]
                if (i - 12) % 4 > 1
                else [0, 0, 0],
                [-1e-8, -1e-8, -1e-8],
            ]
            for i in range((n_joints_body+1)+4*4)
        ],
        'drives_init': [2, 0],
        'weight_osc_body': 1e1,
        'weight_osc_legs_internal': 3e1,
        'weight_osc_legs_opposite': 1e0,  # 1e1,
        'weight_osc_legs_following': 0,  # 1e1,
        'weight_osc_legs2body': 3e1,
        'weight_sens_contact_intralimb': -1e-6,
        'weight_sens_contact_opposite': +1e-6,
        'weight_sens_contact_following': 0,
        'weight_sens_contact_diagonal': 0,
        'weight_sens_hydro_freq': 0,
        'weight_sens_hydro_amp': 0,
        'body_stand_amplitude': 0.2,
        'body_stand_shift': np.pi/2,
        'legs_amplitudes': [np.pi/4, np.pi/32, np.pi/4, np.pi/8],
        'legs_offsets_walking': [0, np.pi/32, 0, np.pi/8],
        'modular_phases': np.array([3*np.pi/2, 0, 3*np.pi/2, 0]) - np.pi/4,
        # 'modular_amplitudes': np.full(4, 1.0),
        'modular_amplitudes': np.full(4, 0),
        'default_lateral_friction': 1.0,
        # Timestep: 1e-3 [s]
        # 'muscle_alpha': 3e-3,
        # 'muscle_beta': -1e-6,
        # 'muscle_gamma': 5e3,
        # 'muscle_delta': -1e-8,
        # 'muscle_alpha': 2e-3,
        # 'muscle_beta': -1e-6,
        # 'muscle_gamma': 2e3,
        # 'muscle_delta': -1e-8,
        'muscle_alpha': 1e-3,
        'muscle_beta': -1e-6,
        'muscle_gamma': 2e3,
        'muscle_delta': -1e-8,
        # # Timestep: 1e-3 [s] (Full scale)
        # 'muscle_alpha': 2e0,
        # 'muscle_beta': -1e0,
        # 'muscle_gamma': 3e0,
        # 'muscle_delta': -1e-3,
        # # Timestep: 1e-2 [s] NOT WORKING
        # 'muscle_alpha': 1e-1,
        # 'muscle_beta': -1e-2,
        # 'muscle_gamma': 3e0,
        # 'muscle_delta': -1e-6,
    }
    kwargs_options.update(kwargs)
    return kwargs_options


def get_salamander_options(**kwargs):
    """Salamander options"""
    kwargs_options = get_salamander_kwargs_options(**kwargs)
    options = get_animat_options(**kwargs_options)
    for joint_i, joint in enumerate(options['morphology']['joints']):
        joint['pybullet_dynamics']['jointDamping'] = 0
        joint['pybullet_dynamics']['maxJointVelocity'] = np.inf  # 0.1
        # joint['pybullet_dynamics']['jointLowerLimit'] = -1e8  # -0.1
        # joint['pybullet_dynamics']['jointUpperLimit'] = +1e8  # +0.1
        joint['pybullet_dynamics']['jointLimitForce'] = np.inf
        joint_control = options['control']['joints'][joint_i]
        assert joint['name'] == joint_control['joint']
        joint['initial_position'] = joint_control['bias']
        # print('{}: {} [rad]'.format(joint['name'], joint_control['bias']))
    return options


def get_centipede_kwargs_options(**kwargs):
    """Centipede options"""
    n_joints_body = kwargs.pop('n_joints_body', 20)
    kwargs_options = {
        'spawn_loader': SpawnLoader.FARMS,  # SpawnLoader.PYBULLET,
        'spawn_position': [0, 0, 0.2*0.07],
        'spawn_orientation': [0, 0, 0],
        'use_self_collisions': False,
        'default_control_type': ControlType.POSITION,  # ControlType.TORQUE,
        'show_hydrodynamics': True,
        'swimming': False,
        'n_legs': 2*19,
        'n_dof_legs': 4,
        'n_joints_body': n_joints_body,
        'drag_coefficients': [
            [
                [-1e-1, -1e0, -1e0]
                if i < 12
                else [-1e-4, -1e-4, -1e-4]
                if (i - 12) % 4 > 1
                else [0, 0, 0],
                [-1e-8, -1e-8, -1e-8],
            ]
            for i in range((n_joints_body+1)+2*4*19)
        ],
        'drives_init': [2, 0],
        'weight_osc_body': 1e1,
        'weight_osc_legs_internal': 3e1,
        'weight_osc_legs_opposite': 1e0,  # 1e1,
        'weight_osc_legs_following': 0,  # 1e1,
        'weight_osc_legs2body': 3e1,
        'weight_sens_contact_intralimb': -1e-6,
        'weight_sens_contact_opposite': +1e-6,
        'weight_sens_contact_following': 0,
        'weight_sens_contact_diagonal': 0,
        'weight_sens_hydro_freq': 0,
        'weight_sens_hydro_amp': 0,
        'body_stand_amplitude': 0.2,
        'body_stand_shift': np.pi/2,
        'legs_amplitudes': [np.pi/4, np.pi/32, np.pi/4, np.pi/8],
        'legs_offsets_walking': [0, np.pi/32, 0, np.pi/8],
        'modular_phases': np.array([3*np.pi/2, 0, 3*np.pi/2, 0]) - np.pi/4,
        # 'modular_amplitudes': np.full(4, 1.0),
        'modular_amplitudes': np.full(4, 0),
        'default_lateral_friction': 1.0,
        # Timestep: 1e-3 [s]
        # 'muscle_alpha': 3e-3,
        # 'muscle_beta': -1e-6,
        # 'muscle_gamma': 5e3,
        # 'muscle_delta': -1e-8,
        # 'muscle_alpha': 2e-3,
        # 'muscle_beta': -1e-6,
        # 'muscle_gamma': 2e3,
        # 'muscle_delta': -1e-8,
        'muscle_alpha': 1e-3,
        'muscle_beta': -1e-6,
        'muscle_gamma': 2e3,
        'muscle_delta': -1e-8,
        # # Timestep: 1e-3 [s] (Full scale)
        # 'muscle_alpha': 2e0,
        # 'muscle_beta': -1e0,
        # 'muscle_gamma': 3e0,
        # 'muscle_delta': -1e-3,
        # # Timestep: 1e-2 [s] NOT WORKING
        # 'muscle_alpha': 1e-1,
        # 'muscle_beta': -1e-2,
        # 'muscle_gamma': 3e0,
        # 'muscle_delta': -1e-6,
    }
    kwargs_options.update(kwargs)
    return kwargs_options


def get_centipede_options(**kwargs):
    """Centipede options"""
    kwargs_options = get_centipede_kwargs_options(**kwargs)
    options = get_animat_options(**kwargs_options)
    for joint_i, joint in enumerate(options['morphology']['joints']):
        joint['pybullet_dynamics']['jointDamping'] = 0
        joint['pybullet_dynamics']['maxJointVelocity'] = np.inf  # 0.1
        # joint['pybullet_dynamics']['jointLowerLimit'] = -1e8  # -0.1
        # joint['pybullet_dynamics']['jointUpperLimit'] = +1e8  # +0.1
        joint['pybullet_dynamics']['jointLimitForce'] = np.inf
        joint_control = options['control']['joints'][joint_i]
        assert joint['name'] == joint_control['joint']
        joint['initial_position'] = joint_control['bias']
        # print('{}: {} [rad]'.format(joint['name'], joint_control['bias']))
    return options


def get_pleurobot_kwargs_options(**kwargs):
    """Pleurobot default options"""

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
    links_swimming = links_names[:14]
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
    # links_no_collisions = kwargs.pop('links_no_collisions', [
    #     link
    #     for link in links_names
    #     if link not in feet+['Head', 'link_tail']
    # ])
    links_no_collisions = kwargs.pop('links_no_collisions', [])

    # Joint options
    gain_amplitude = kwargs.pop('gain_amplitude', None)
    joints_offsets = kwargs.pop('joints_offsets', None)

    # Amplitudes gains
    if gain_amplitude is None:
        gain_amplitude = [-1]*(13+4*4)  # np.ones(13+4*4)
        gain_amplitude[6] = 0
        gain_amplitude[12] = 0
        for leg_i in range(2):
            for side_i in range(2):
                mirror = (-1 if side_i else 1)
                gain_amplitude[13+2*leg_i*4+side_i*4+0] = mirror
                gain_amplitude[13+2*leg_i*4+side_i*4+1] = mirror
                gain_amplitude[13+2*leg_i*4+side_i*4+2] = -mirror
                gain_amplitude[13+2*leg_i*4+side_i*4+3] = mirror
        gain_amplitude = dict(zip(joints_names, gain_amplitude))

    # Joints joints_offsets
    if joints_offsets is None:
        joints_offsets = [0]*(13+4*4)
        for leg_i in range(2):
            for side_i in range(2):
                mirror = (1 if side_i else -1)
                joints_offsets[13+2*leg_i*4+side_i*4+0] = 0
                joints_offsets[13+2*leg_i*4+side_i*4+1] = (
                    mirror*np.pi/16 if leg_i else mirror*np.pi/8
                )
                joints_offsets[13+2*leg_i*4+side_i*4+2] = (
                    0 if leg_i else -mirror*np.pi/3
                )
                joints_offsets[13+2*leg_i*4+side_i*4+3] = (
                    -mirror*np.pi/4 if leg_i else mirror*np.pi/16
                )
        joints_offsets = dict(zip(joints_names, joints_offsets))

    # Animat options
    kwargs_options = dict(
        spawn_loader=SpawnLoader.PYBULLET,  # SpawnLoader.FARMS,  # SpawnLoader.PYBULLET,
        default_control_type=ControlType.POSITION,
        swimming=False,
        n_legs=4,
        n_dof_legs=4,
        n_joints_body=13,
        use_self_collisions=True,
        drag_coefficients=[
            [
                [-1e1, -1e1, -1e1]
                if link in links_swimming
                else [-0, -0, -0],
                [0, 0, 0],
            ]
            for link in links_names
        ],
        links_names=links_names,
        links_swimming=links_swimming,
        links_no_collisions=links_no_collisions,
        joints_names=joints_names,
        sensors_gps=links_names,
        sensors_joints=joints_names,
        sensors_contacts=feet,
        sensors_hydrodynamics=[],
        default_lateral_friction=2,
    )
    if 'kinematics_file' not in kwargs:
        kwargs_options.update(dict(
            body_stand_amplitude=0.2,
            legs_amplitudes=[np.pi/8, np.pi/16, np.pi/8, np.pi/8],
            legs_offsets_walking=[0, -np.pi/32, -np.pi/16, 0],
            legs_offsets_swimming=[-2*np.pi/5, 0, 0, -np.pi/4],
            gain_amplitude=gain_amplitude,
            offsets_bias=joints_offsets,
            weight_osc_body=1e0,
            weight_osc_legs_internal=3e1,
            weight_osc_legs_opposite=1e0,
            weight_osc_legs_following=5e-1,
            weight_osc_legs2body=3e1,
            weight_sens_contact_intralimb=-2e-1,
            weight_sens_contact_opposite=5e-1,
            weight_sens_contact_following=0,
            weight_sens_contact_diagonal=0,
            weight_sens_hydro_freq=0,
            weight_sens_hydro_amp=0,
            modular_phases=np.array([3*np.pi/2, 0, 3*np.pi/2, 0]) - np.pi/4,
            modular_amplitudes=np.full(4, 0.9),
            muscle_alpha=5e1,
            muscle_beta=-1e1,
            muscle_gamma=1e1,
            muscle_delta=-3e-1,
        ))
    kwargs_options.update(kwargs)
    return kwargs_options


def get_pleurobot_options(**kwargs):
    """Pleurobot default options"""

    # Animat
    sdf = get_sdf_path(name='pleurobot', version='1')
    pylog.info('Model SDF: {}'.format(sdf))

    kwargs_options = get_pleurobot_kwargs_options(**kwargs)
    animat_options = get_animat_options(**kwargs_options)
    return sdf, animat_options


def get_krock_kwargs_options(**kwargs):
    """Krock default options"""

    # Morphology information
    # links_all = [
    #     'base_link', 'krock2', 'Head_T', 'solid_head',
    #     'Head_transform', 'FrontGirdle', 'FrontGirdleSpineMX',
    #     'FrontGirdleMain', 'FrontLeftPitchMX', 'FrontRightPitchMX', 'SPINE1_T',
    #     'solid_spine1_endpoint', 'solid_spine1_endpoint_C', 'SPINE2_T',
    #     'solid_spine2_endpoint', 'Tail1MX', 'TAIL1_T', 'solid_tail1_endpoint',
    #     'TAIL2_T', 'solid_tail1_passive_endpoint', 'TAIL3_T',
    #     'solid_tail3_endpoint', 'Fin_Transform', 'TAIL2MX', 'Tail2MX',
    #     'HLpitch_T', 'solid_hlpitch_hj_endpoint', 'HLyaw_HJ_C', 'HLroll_T',
    #     'HLroll_HJ_C', 'HLknee_HJ_C', 'HL_TOUCH_T', 'TS_HL', 'LeftHindFoot',
    #     'HindLeftKneeMX', 'HindLeftRollMX', 'HindLeftFemur', 'HindLeftYawMX',
    #     'HRpitch_T', 'HRpitch_HJ_C', 'HRyaw_HJ_C', 'HRroll_T', 'HRroll_HJ_C',
    #     'HRknee_HJ_C', 'HR_TOUCH_T', 'TS_HR', 'RightHindFoot',
    #     'HindRightKneeMX', 'HindRightRollMX', 'HindRightFemur',
    #     'HindRightYawMX', 'HindLeftPitchMX', 'HindRightPitchMX',
    #     'HindGirdleSpineMX', 'HindGirdleMain', 'SPINE_Transform', 'FLpitch_T',
    #     'solid_flpitch_hj_endpoint', 'solid_flyaw_hj_endpoint', 'FLroll_T',
    #     'FLroll_HJ_C', 'FLknee_HJ_C', 'FL_TOUCH_T', 'TS_FL', 'LeftFrontFoot',
    #     'FrontLeftKneeMX', 'FrontLeftRollMX', 'FrontLeftFemur',
    #     'FrontLeftYawMX', 'FRpitch_T', 'FRpitch_HJ_C', 'FRyaw_HJ_C', 'FRroll_T',
    #     'FRroll_HJ_C', 'FRknee_HJ_C', 'FR_TOUCH_T', 'TS_FR', 'RightFrontFoot',
    #     'FrontRightKneeMX', 'FrontRightRollMX', 'FrontRightFemur',
    #     'FrontRightYawMX',
    # ]
    # joints_all = [
    #     'solid_spine1_endpoint_to_SPINE1_T_SPINE1_HJ',
    #     'solid_spine2_endpoint_to_SPINE2_T_SPINE2_HJ',
    #     'solid_tail1_endpoint_to_TAIL1_T_TAIL1_T_C',
    #     'solid_tail1_passive_endpoint_to_TAIL2_T_TAIL2_T_C',
    #     'solid_tail3_endpoint_to_TAIL3_T_TAIL3_T_C',
    #     'solid_hlpitch_hj_endpoint_to_HLpitch_T_HLpitch_HJ',
    #     'HLyaw_HJ_C_to_solid_hlpitch_hj_endpoint_HLyaw_HJ',
    #     'HLroll_HJ_C_to_HLroll_T_HLroll_HJ',
    #     'HLknee_HJ_C_to_HLroll_HJ_C_HLknee_HJ',
    #     'HRpitch_HJ_C_to_HRpitch_T_HRpitch_HJ',
    #     'HRyaw_HJ_C_to_HRpitch_HJ_C_HRyaw_HJ',
    #     'HRroll_HJ_C_to_HRroll_T_HRroll_HJ',
    #     'HRknee_HJ_C_to_HRroll_HJ_C_HRknee_HJ',
    #     'solid_flpitch_hj_endpoint_to_FLpitch_T_FLpitch_HJ',
    #     'solid_flyaw_hj_endpoint_to_solid_flpitch_hj_endpoint_FLyaw_HJ',
    #     'FLroll_HJ_C_to_FLroll_T_FLroll_HJ',
    #     'FLknee_HJ_C_to_FLroll_HJ_C_FLknee_HJ',
    #     'FRpitch_HJ_C_to_FRpitch_T_FRpitch_HJ',
    #     'FRyaw_HJ_C_to_FRpitch_HJ_C_FRyaw_HJ',
    #     'FRroll_HJ_C_to_FRroll_T_FRroll_HJ',
    #     'FRknee_HJ_C_to_FRroll_HJ_C_FRknee_HJ',
    # ]
    links_inertials = [
        'krock2',
        'solid_head',
        'solid_spine1_endpoint',
        'solid_spine2_endpoint',
        'solid_tail1_endpoint',
        'solid_tail1_passive_endpoint',
        'solid_tail3_endpoint',
        'solid_hlpitch_hj_endpoint',
        'HLyaw_HJ_C',
        'HLroll_HJ_C',
        'HLknee_HJ_C',
        'TS_HL',
        'HRpitch_HJ_C',
        'HRyaw_HJ_C',
        'HRroll_HJ_C',
        'HRknee_HJ_C',
        'TS_HR',
        'solid_flpitch_hj_endpoint',
        'solid_flyaw_hj_endpoint',
        'FLroll_HJ_C',
        'FLknee_HJ_C',
        'TS_FL',
        'FRpitch_HJ_C',
        'FRyaw_HJ_C',
        'FRroll_HJ_C',
        'FRknee_HJ_C',
        'TS_FR',
    ]
    links_names = kwargs.pop(
        'links_names',
        [
            # Body
            'base_link',
            'krock2',
            'solid_spine1_endpoint',
            'Tail1MX',
            'solid_tail1_endpoint',
            'solid_tail1_passive_endpoint',
            'solid_tail3_endpoint',
            # Limb (FL)
            'FRpitch_HJ_C',
            'FRyaw_HJ_C',
            'FRroll_HJ_C',
            # 'FRknee_HJ_C',
            'TS_FR',
            # Limb (FR)
            'FLroll_T',
            'FLroll_HJ_C',
            'FLknee_HJ_C',
            # 'FL_TOUCH_T',
            'TS_FL',
            # Limb (HL)
            'HRpitch_HJ_C',
            'HRyaw_HJ_C',
            'HRroll_HJ_C',
            # 'HRknee_HJ_C',
            'TS_HR',
            # Limb (HR)
            'HLroll_T',
            'HLyaw_HJ_C',
            'HLroll_HJ_C',
            # 'HLknee_HJ_C',
            'TS_HL',
        ]
    )
    joints_names = kwargs.pop('joints_names', [
        # Body
        'solid_spine1_endpoint_to_SPINE1_T_SPINE1_HJ',
        'solid_spine2_endpoint_to_SPINE2_T_SPINE2_HJ',
        'solid_tail1_endpoint_to_TAIL1_T_TAIL1_T_C',
        'solid_tail1_passive_endpoint_to_TAIL2_T_TAIL2_T_C',
        'solid_tail3_endpoint_to_TAIL3_T_TAIL3_T_C',
        # Limb (FL)
        'FRpitch_HJ_C_to_FRpitch_T_FRpitch_HJ',
        'FRyaw_HJ_C_to_FRpitch_HJ_C_FRyaw_HJ',
        'FRroll_HJ_C_to_FRroll_T_FRroll_HJ',
        'FRknee_HJ_C_to_FRroll_HJ_C_FRknee_HJ',
        # Limb (FR)
        'solid_flpitch_hj_endpoint_to_FLpitch_T_FLpitch_HJ',
        'solid_flyaw_hj_endpoint_to_solid_flpitch_hj_endpoint_FLyaw_HJ',
        'FLroll_HJ_C_to_FLroll_T_FLroll_HJ',
        'FLknee_HJ_C_to_FLroll_HJ_C_FLknee_HJ',
        # Limb (HL)
        'HRpitch_HJ_C_to_HRpitch_T_HRpitch_HJ',
        'HRyaw_HJ_C_to_HRpitch_HJ_C_HRyaw_HJ',
        'HRroll_HJ_C_to_HRroll_T_HRroll_HJ',
        'HRknee_HJ_C_to_HRroll_HJ_C_HRknee_HJ',
        # Limb (HR)
        'solid_hlpitch_hj_endpoint_to_HLpitch_T_HLpitch_HJ',
        'HLyaw_HJ_C_to_solid_hlpitch_hj_endpoint_HLyaw_HJ',
        'HLroll_HJ_C_to_HLroll_T_HLroll_HJ',
        'HLknee_HJ_C_to_HLroll_HJ_C_HLknee_HJ',
    ])
    feet = kwargs.pop('feet', ['TS_FR', 'TS_FL', 'TS_HR', 'TS_HL'])
    # links_no_collisions = kwargs.pop('links_no_collisions', [
    #     link
    #     for link in links_names
    #     if link not in feet+['Head', 'link_tail']
    # ])
    links_no_collisions = kwargs.pop('links_no_collisions', [])

    # Joint options
    gain_amplitude = kwargs.pop('gain_amplitude', None)
    joints_offsets = kwargs.pop('joints_offsets', None)

    # Amplitudes gains
    if gain_amplitude is None:
        gain_amplitude = [1]*(5+4*4)  # np.ones(5+4*4)
        # gain_amplitude[5] = 0
        for leg_i in range(2):
            for side_i in range(2):
                mirror = (1 if side_i else -1)
                mirror2 = (-1 if leg_i else 1)
                gain_amplitude[5+2*leg_i*4+side_i*4+0] = mirror
                gain_amplitude[5+2*leg_i*4+side_i*4+1] = mirror  # mirror
                gain_amplitude[5+2*leg_i*4+side_i*4+2] = 1  # -mirror
                gain_amplitude[5+2*leg_i*4+side_i*4+3] = 0  # mirror
        gain_amplitude = dict(zip(joints_names, gain_amplitude))

    # Joints joints_offsets
    if joints_offsets is None:
        joints_offsets = [0]*(5+4*4)
        for leg_i in range(2):
            for side_i in range(2):
                mirror = (1 if side_i else -1)
                mirror2 = (-1 if leg_i else 1)
                joints_offsets[5+2*leg_i*4+side_i*4+0] = -0.1*np.pi*mirror
                joints_offsets[5+2*leg_i*4+side_i*4+1] = 0*np.pi*mirror  # (
                #     mirror*np.pi/16 if leg_i else mirror*np.pi/8
                # )
                joints_offsets[5+2*leg_i*4+side_i*4+2] = 0.5*np.pi*mirror2
                #     0 if leg_i else -mirror*np.pi/3
                # )
                joints_offsets[5+2*leg_i*4+side_i*4+3] = 0  # (
                #     -mirror*np.pi/4 if leg_i else mirror*np.pi/16
                # )
        joints_offsets = dict(zip(joints_names, joints_offsets))

    # Animat options
    kwargs_options = dict(
        spawn_loader=SpawnLoader.PYBULLET,  # SpawnLoader.FARMS,
        density=300.0,
        # mass_multiplier=0.7,
        spawn_position=[0, 0, 0.5],
        spawn_orientation=[-0.5*np.pi, 0, np.pi],
        default_control_type=ControlType.POSITION,
        swimming=False,
        show_hydrodynamics=True,
        n_legs=4,
        n_dof_legs=4,
        n_links_body=7,
        n_joints_body=5,
        use_self_collisions=False,
        drag_coefficients=[
            [
                [-1e0, -1e0, -1e2]
                if i < 6
                else [-1e0, -1e0, -1e0],
                [-1e-8, -1e-8, -1e-8],
            ]
            for i in range(6+4*4)
        ],
        body_stand_amplitude=0.2,
        legs_amplitudes=[np.pi/16, np.pi/8, np.pi/8, np.pi/8],
        legs_offsets_walking=[0, -np.pi/32, -np.pi/16, 0],
        legs_offsets_swimming=[0, 0.25*np.pi, 0.5*np.pi, 0, np.pi/2],
        gain_amplitude=gain_amplitude,
        offsets_bias=joints_offsets,
        intralimb_phases=[0, 0.5*np.pi, 0.5*np.pi, 0.5*np.pi, 0],
        weight_osc_body=1e0,
        weight_osc_legs_internal=3e1,
        weight_osc_legs_opposite=1e0,
        weight_osc_legs_following=5e-1,
        weight_osc_legs2body=3e1,
        weight_sens_contact_intralimb=0,
        weight_sens_contact_opposite=0,
        weight_sens_contact_following=0,
        weight_sens_contact_diagonal=0,
        weight_sens_hydro_freq=0,
        weight_sens_hydro_amp=0,
        modular_phases=np.array([3*np.pi/2, 0, 3*np.pi/2, 0]) - np.pi/4,
        modular_amplitudes=np.full(4, 0.5),
        links_names=links_names,
        links_swimming=links_inertials,
        links_no_collisions=links_no_collisions,
        joints_names=joints_names,
        sensors_gps=['base_link'] + links_inertials,
        sensors_joints=joints_names,
        sensors_contacts=feet,
        sensors_hydrodynamics=links_inertials,
        default_lateral_friction=2,
        muscle_alpha=5e1,
        muscle_beta=-1e1,
        muscle_gamma=1e1,
        muscle_delta=-3e-1,
    )
    kwargs_options.update(kwargs)
    return kwargs_options


def get_krock_options(**kwargs):
    """Krock default options"""

    # Animat
    sdf = get_sdf_path(name='krock', version='0')
    pylog.info('Model SDF: {}'.format(sdf))

    kwargs_options = get_krock_kwargs_options(**kwargs)
    animat_options = get_animat_options(**kwargs_options)
    return sdf, animat_options


def get_orobot_kwargs_options(**kwargs):
    """Orobot default options"""

    # Morphology information
    links_names = kwargs.pop(
        'links_names',
        [
            # # Body
            # 'base_link',
            # 'OROBOT',
            # 'PECTORAL_GIRDLE2',
            # 'S_SPINE3_C',
            # 'S_SPINE4_C',
            # 'S_SPINE5_C',
            # 'HIND_GIRDLE_C',
            # 'SPINE6_BACK_TT_C',
            # 'S_SPINE8_C',
            # # Limb (FL)
            # 'LEFT_FRONT_YAW_C',
            # 'LEFT_FRONT_PITCH_C',
            # # 'LEFT_FRONT_ELBOW_C',
            # 'LEFT_ULNA_T_C',
            # # Limb (FR)
            # 'RIGHT_FRONT_YAW_C',
            # 'RIGHT_FRONT_PITCH_C',
            # # 'RIGHT_FRONT_ELBOW_C',
            # 'RIGHT_ULNA_T_C',
            # # Limb (HL)
            # 'LEFT_HIP_PITCH_C',
            # 'LEFT_HIND_ROLL_T_C',
            # # 'LEFT_KNEE_T_C',
            # 'LEFT_CRUS_T_C',
            # # Limb (HR)
            # 'RIGHT_HIP_PITCH_C',
            # 'RIGHT_HIND_ROLL_T_C',
            # # 'RIGHT_KNEE_T_C',
            # 'RIGHT_CRUS_T_C',

            # Body
            'base_link',
            'OROBOT',
            # 'HEAD_TT',
            # 'HEAD_TT_C',
            # 'LOWER_JAW_TT',
            # 'LOWER_JAW_TT_C',
            # 'S_SPINE1T',
            # 'S_SPINE1_C',
            # 'PECTORAL_GIRDLE1',
            # 'RX28_T',
            # 'SPINE1_TT',
            # 'SPINE1_TT_C',
            # 'S_SPINE2T',
            'S_SPINE2_C',
            # 'PECTORAL_GIRDLE2',

            # 'MOTOR_TT1',
            # 'MOTOR_TT1_C',
            # 'SPINE2_TT',
            # 'SPINE2_TT_C',
            # 'S_SPINE3T',
            'S_SPINE3_C',
            # 'MOTOR_TT2',
            # 'MOTOR_TT2_C',
            # 'SPINE3_TT',
            # 'SPINE3_TT_C',
            # 'S_SPINE4T',
            'S_SPINE4_C',
            # 'S_SPINE4_C_C',
            # 'S_SPINE4_C_C_C',
            # 'SPINE4_TT',
            # 'SPINE4_TT_C',
            # 'S_SPINE5T',
            'S_SPINE5_C',
            # 'S_SPINE5_C_C',
            # 'S_SPINE5_C_C_C',
            # 'SPINE5_TT',
            # 'SPINE5_TT_C',
            # 'S_SPINE6T',
            'S_SPINE6_C',
            # 'HIND_GIRDLE',
            # 'HIND_GIRDLE_C',
            # 'HIND_GIRDLE_C_C',
            # 'SPINE6_FRONT_TT',
            # 'SPINE6_FRONT_TT_C',

            # 'PS_SPINE1T',
            # 'PS_SPINE1_C',
            # 'SPINE6_BACK_TT',
            # 'SPINE6_BACK_TT_C',
            # 'S_SPINE7T',
            'S_SPINE7_C',
            # 'S_SPINE7_C_C',
            # 'S_SPINE7_C_C_C',
            # 'SPINE7_TT',
            # 'SPINE7_TT_C',
            # 'S_SPINE8T',
            'S_SPINE8_C',
            # 'S_SPINE8_C_C',
            # 'S_SPINE8_C_C_C',
            # 'SPINE10_TT',
            # 'SPINE10_TT_C',

            'LEFT_SHOULDER_YAW',
            # 'LEFT_SHOULDER_YAW_C',
            # 'LEFT_FRONT_YAWT',
            # 'LEFT_FRONT_YAW_C',
            'LEFT_SHOULDER_PITCH',
            # 'LEFT_SHOULDER_PITCH_C',
            # 'LEFT_FRONT_PITCHT',
            # 'LEFT_FRONT_PITCH_C',
            'LEFT_HUMERUS_TT',
            # 'LEFT_HUMERUS_TT_C',
            # 'LEFT_ROLL_T',
            # 'LEFT_ROLL_T_C',
            # 'LEFT_FRONT_ROLLT',
            # 'LEFT_FRONT_ROLL_C',
            'LEFT_ELBOW_T',
            # 'LEFT_ELBOW_T_C',
            # 'LEFT_FRONT_ELBOWT',
            'LEFT_FRONT_ELBOW_C',
            # 'LEFT_FRONT_MX28',
            # 'LEFT_FRONT_WRISTT',
            # 'LEFT_FRONT_WRIST_C',
            # 'LEFT_ULNA_T',
            # 'LEFT_ULNA_T_C',

            'RIGHT_SHOULDER_YAW',
            # 'RIGHT_SHOULDER_YAW_C',
            # 'RIGHT_FRONT_YAWT',
            # 'RIGHT_FRONT_YAW_C',
            'RIGHT_SHOULDER_PITCH',
            # 'RIGHT_SHOULDER_PITCH_C',
            # 'RIGHT_FRONT_PITCHT',
            # 'RIGHT_FRONT_PITCH_C',
            'RIGHT_HUMERUS_TT',
            # 'RIGHT_HUMERUS_TT_C',
            # 'RIGHT_ROLL_T',
            # 'RIGHT_ROLL_T_C',
            # 'RIGHT_FRONT_ROLLT',
            # 'RIGHT_FRONT_ROLL_C',
            'RIGHT_ELBOW_T',
            # 'RIGHT_ELBOW_T_C',
            # 'RIGHT_FRONT_ELBOWT',
            'RIGHT_FRONT_ELBOW_C',
            # 'RIGHT_FRONT_MX28',
            # 'RIGHT_FRONT_WRISTT',
            # 'RIGHT_FRONT_WRIST_C',
            # 'RIGHT_ULNA_T',
            # 'RIGHT_ULNA_T_C',

            'LEFT_HIP_TT',
            # 'LEFT_HIP_TT_C',
            # 'LEFT_HIP_TT_C_C',
            # 'LEFT_HIND_YAWT',
            # 'LEFT_HIND_YAW_C',
            'LEFT_HIP_PITCH',
            # 'LEFT_HIP_PITCH_C',
            # 'LEFT_HIND_PITCHT',
            # 'LEFT_HIND_PITCH_C',
            'LEFT_FEMUR_TT',
            # 'LEFT_FEMUR_TT_C',
            # 'LEFT_HIND_ROLL_T',
            # 'LEFT_HIND_ROLL_T_C',
            # 'LEFT_HIND_ROLLT',
            # 'LEFT_HIND_ROLL_C',
            'LEFT_KNEE_T',
            # 'LEFT_KNEE_T_C',
            # 'LEFT_HIND_KNEET',
            'LEFT_HIND_KNEE_C',
            # 'LEFT_HIND_MX28',
            # 'LEFT_HIND_WRISTT',
            # 'LEFT_HIND_WRIST_C',
            # 'LEFT_CRUS_T',
            # 'LEFT_CRUS_T_C',

            'RIGHT_HIP_TT',
            # 'RIGHT_HIP_TT_C',
            # 'RIGHT_HIP_TT_C_C',
            # 'RIGHT_HIND_YAWT',
            # 'RIGHT_HIND_YAW_C',
            'RIGHT_HIP_PITCH',
            # 'RIGHT_HIP_PITCH_C',
            # 'RIGHT_HIND_PITCHT',
            # 'RIGHT_HIND_PITCH_C',
            'RIGHT_FEMUR_TT',
            # 'RIGHT_FEMUR_TT_C',
            # 'RIGHT_HIND_ROLL_T',
            # 'RIGHT_HIND_ROLL_T_C',
            # 'RIGHT_HIND_ROLLT',
            # 'RIGHT_HIND_ROLL_C',
            'RIGHT_KNEE_T',
            # 'RIGHT_KNEE_T_C',
            # 'RIGHT_HIND_KNEET',
            'RIGHT_HIND_KNEE_C',
            # 'RIGHT_HIND_MX28',
            # 'RIGHT_HIND_WRISTT',
            # 'RIGHT_HIND_WRIST_C',
            # 'RIGHT_CRUS_T',
            # 'RIGHT_CRUS_T_C',
        ]
    )
    joints_names = kwargs.pop('joints_names', [
        # # Body
        # 'S_SPINE1_C_to_S_SPINE1T_S_SPINE1',
        # 'S_SPINE2_C_to_S_SPINE2T_S_SPINE2',
        # 'S_SPINE3_C_to_S_SPINE3T_S_SPINE3',
        # 'S_SPINE4_C_to_S_SPINE4T_S_SPINE4',
        # 'S_SPINE5_C_to_S_SPINE5T_S_SPINE5',
        # 'S_SPINE6_C_to_S_SPINE6T_S_SPINE6',
        # 'S_SPINE7_C_to_S_SPINE7T_S_SPINE7',
        # 'S_SPINE8_C_to_S_SPINE8T_S_SPINE8',
        # # Limb (FL)
        # 'LEFT_FRONT_YAW_C_to_LEFT_FRONT_YAWT_LEFT_FRONT_YAW',
        # 'LEFT_FRONT_PITCH_C_to_LEFT_FRONT_PITCHT_LEFT_FRONT_PITCH',
        # 'LEFT_FRONT_WRIST_C_to_LEFT_FRONT_WRISTT_LEFT_FRONT_WRIST',
        # # 'LEFT_FRONT_ELBOW_C_to_LEFT_FRONT_ELBOWT_LEFT_FRONT_ELBOW',
        # # Limb (FR)
        # 'RIGHT_FRONT_YAW_C_to_RIGHT_FRONT_YAWT_RIGHT_FRONT_YAW',
        # 'RIGHT_FRONT_PITCH_C_to_RIGHT_FRONT_PITCHT_RIGHT_FRONT_PITCH',
        # 'RIGHT_FRONT_WRIST_C_to_RIGHT_FRONT_WRISTT_RIGHT_FRONT_WRIST',
        # # 'RIGHT_FRONT_ELBOW_C_to_RIGHT_FRONT_ELBOWT_RIGHT_FRONT_ELBOW',
        # # Limb (HL)
        # 'LEFT_HIND_YAW_C_to_LEFT_HIND_YAWT_LEFT_HIND_YAW',
        # 'LEFT_HIND_PITCH_C_to_LEFT_HIND_PITCHT_LEFT_HIND_PITCH',
        # # 'LEFT_HIND_WRIST_C_to_LEFT_HIND_WRISTT_LEFT_HIND_WRIST',
        # 'LEFT_HIND_KNEE_C_to_LEFT_HIND_KNEET_LEFT_HIND_KNEE',
        # # Limb (HR)
        # 'RIGHT_HIND_YAW_C_to_RIGHT_HIND_YAWT_RIGHT_HIND_YAW',
        # 'RIGHT_HIND_PITCH_C_to_RIGHT_HIND_PITCHT_RIGHT_HIND_PITCH',
        # # 'RIGHT_HIND_WRIST_C_to_RIGHT_HIND_WRISTT_RIGHT_HIND_WRIST',
        # # 'RIGHT_HIND_MX28_to_RIGHT_HIND_KNEE_C',
        # 'RIGHT_HIND_KNEE_C_to_RIGHT_HIND_KNEET_RIGHT_HIND_KNEE',


        # Body
        'S_SPINE1_C_to_S_SPINE1T_S_SPINE1',
        'S_SPINE2_C_to_S_SPINE2T_S_SPINE2',
        'S_SPINE3_C_to_S_SPINE3T_S_SPINE3',
        'S_SPINE4_C_to_S_SPINE4T_S_SPINE4',
        'S_SPINE5_C_to_S_SPINE5T_S_SPINE5',
        'S_SPINE6_C_to_S_SPINE6T_S_SPINE6',
        'S_SPINE7_C_to_S_SPINE7T_S_SPINE7',
        'S_SPINE8_C_to_S_SPINE8T_S_SPINE8',
        # 'PS_SPINE1_C_to_PS_SPINE1T_PS_SPINE1',  # Passive
        # Limb (FL)
        'LEFT_FRONT_YAW_C_to_LEFT_FRONT_YAWT_LEFT_FRONT_YAW',
        'LEFT_FRONT_PITCH_C_to_LEFT_FRONT_PITCHT_LEFT_FRONT_PITCH',
        'LEFT_FRONT_ROLL_C_to_LEFT_FRONT_ROLLT_LEFT_FRONT_ROLL',
        'LEFT_FRONT_WRIST_C_to_LEFT_FRONT_WRISTT_LEFT_FRONT_WRIST',
        'LEFT_FRONT_ELBOW_C_to_LEFT_FRONT_ELBOWT_LEFT_FRONT_ELBOW',
        # Limb (FR)
        'RIGHT_FRONT_YAW_C_to_RIGHT_FRONT_YAWT_RIGHT_FRONT_YAW',
        'RIGHT_FRONT_PITCH_C_to_RIGHT_FRONT_PITCHT_RIGHT_FRONT_PITCH',
        'RIGHT_FRONT_ROLL_C_to_RIGHT_FRONT_ROLLT_RIGHT_FRONT_ROLL',
        'RIGHT_FRONT_WRIST_C_to_RIGHT_FRONT_WRISTT_RIGHT_FRONT_WRIST',
        'RIGHT_FRONT_ELBOW_C_to_RIGHT_FRONT_ELBOWT_RIGHT_FRONT_ELBOW',
        # Limb (HL)
        'LEFT_HIND_YAW_C_to_LEFT_HIND_YAWT_LEFT_HIND_YAW',
        'LEFT_HIND_PITCH_C_to_LEFT_HIND_PITCHT_LEFT_HIND_PITCH',
        'LEFT_HIND_ROLL_C_to_LEFT_HIND_ROLLT_LEFT_HIND_ROLL',
        'LEFT_HIND_KNEE_C_to_LEFT_HIND_KNEET_LEFT_HIND_KNEE',
        'LEFT_HIND_WRIST_C_to_LEFT_HIND_WRISTT_LEFT_HIND_WRIST',
        # Limb (HR)
        'RIGHT_HIND_YAW_C_to_RIGHT_HIND_YAWT_RIGHT_HIND_YAW',
        'RIGHT_HIND_PITCH_C_to_RIGHT_HIND_PITCHT_RIGHT_HIND_PITCH',
        'RIGHT_HIND_ROLL_C_to_RIGHT_HIND_ROLLT_RIGHT_HIND_ROLL',
        'RIGHT_HIND_KNEE_C_to_RIGHT_HIND_KNEET_RIGHT_HIND_KNEE',
        'RIGHT_HIND_WRIST_C_to_RIGHT_HIND_WRISTT_RIGHT_HIND_WRIST',
    ])
    feet = kwargs.pop(
        'feet',
        [
            # 'LEFT_ULNA_T_C',
            # 'RIGHT_ULNA_T_C',
            # 'LEFT_CRUS_T_C',
            # 'RIGHT_CRUS_T_C',
            'LEFT_FRONT_ELBOW_C',
            'RIGHT_FRONT_ELBOW_C',
            'LEFT_HIND_KNEE_C',
            'RIGHT_HIND_KNEE_C',
        ]
    )
    # links_no_collisions = kwargs.pop('links_no_collisions', [
    #     link
    #     for link in links_names
    #     if link not in feet+['Head', 'link_tail']
    # ])
    links_no_collisions = kwargs.pop('links_no_collisions', [])

    # Joint options
    gain_amplitude = kwargs.pop('gain_amplitude', None)
    joints_offsets = kwargs.pop('joints_offsets', None)

    # Amplitudes gains
    if gain_amplitude is None:
        gain_amplitude = [-1]*(8+4*5)  # np.ones(8+4*5)
        # gain_amplitude[8] = 0
        for leg_i in range(2):
            for side_i in range(2):
                mirror = (1 if side_i else -1)
                # mirror2 = (-1 if leg_i else 1)
                gain_amplitude[8+2*leg_i*5+side_i*5+0] = mirror
                gain_amplitude[8+2*leg_i*5+side_i*5+1] = -mirror
                gain_amplitude[8+2*leg_i*5+side_i*5+2] = -1
                gain_amplitude[8+2*leg_i*5+side_i*5+3] = -0*mirror
                gain_amplitude[8+2*leg_i*5+side_i*5+4] = 0*mirror
        gain_amplitude = dict(zip(joints_names, gain_amplitude))

    # Joints joints_offsets
    if joints_offsets is None:
        joints_offsets = [0]*(8+4*5)
        for leg_i in range(2):
            for side_i in range(2):
                mirror = (1 if side_i else -1)
                # mirror2 = (-1 if leg_i else 1)
                joints_offsets[8+2*leg_i*5+side_i*5+0] = -0*mirror
                joints_offsets[8+2*leg_i*5+side_i*5+1] = 0*mirror
                joints_offsets[8+2*leg_i*5+side_i*5+2] = -0*np.pi
                joints_offsets[8+2*leg_i*5+side_i*5+3] = -0.25*np.pi*mirror
                joints_offsets[8+2*leg_i*5+side_i*5+4] = 0*mirror
        joints_offsets = dict(zip(joints_names, joints_offsets))

    # Animat options
    kwargs_options = dict(
        spawn_loader=SpawnLoader.PYBULLET,  # SpawnLoader.FARMS,
        spawn_position=[0, 0, 0],
        spawn_orientation=[0.5*np.pi, 0, 0],
        default_control_type=ControlType.POSITION,
        swimming=False,
        n_legs=4,
        n_dof_legs=5,
        n_joints_body=8,
        use_self_collisions=True,
        body_stand_amplitude=0.2,
        legs_amplitudes=[np.pi/6, np.pi/16, np.pi/16, np.pi/8, np.pi/8],
        legs_offsets_walking=[0, -np.pi/32, -np.pi/16, 0, 0],
        legs_offsets_swimming=[2*np.pi/5, 0, 0, np.pi/2, 0],
        gain_amplitude=gain_amplitude,
        offsets_bias=joints_offsets,
        weight_osc_body=1e0,
        weight_osc_legs_internal=3e1,
        weight_osc_legs_opposite=1e0,
        weight_osc_legs_following=5e-1,
        weight_osc_legs2body=3e1,
        weight_sens_contact_intralimb=0,
        weight_sens_contact_opposite=0,
        weight_sens_contact_following=0,
        weight_sens_contact_diagonal=0,
        weight_sens_hydro_freq=0,
        weight_sens_hydro_amp=0,
        modular_phases=np.array([3*np.pi/2, 0, 3*np.pi/2, 0, 0]) - np.pi/4,
        modular_amplitudes=np.full(5, 0.5),
        links_names=links_names,
        links_swimming=[],
        links_no_collisions=links_no_collisions,
        joints_names=joints_names,
        sensors_gps=links_names,
        sensors_joints=joints_names,
        sensors_contacts=feet,
        sensors_hydrodynamics=[],
        default_lateral_friction=2,
        muscle_alpha=5e1,
        muscle_beta=-1e1,
        muscle_gamma=1e1,
        muscle_delta=-3e-1,
    )
    kwargs_options.update(kwargs)
    return kwargs_options


def get_orobot_options(**kwargs):
    """Orobot default options"""

    # Animat
    sdf = get_sdf_path(name='orobot', version='0')
    pylog.info('Model SDF: {}'.format(sdf))

    kwargs_options = get_orobot_kwargs_options(**kwargs)
    animat_options = get_animat_options(**kwargs_options)
    return sdf, animat_options
