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


def get_salamander_kwargs_options(**kwargs):
    """Salamander options"""
    kwargs_options = {
        'spawn_loader': SpawnLoader.FARMS,  # SpawnLoader.PYBULLET,
        'default_control_type': ControlType.POSITION,
        'show_hydrodynamics': True,
        'swimming': False,
        'n_legs': 4,
        'n_dof_legs': 4,
        'n_joints_body': 11,
        'use_self_collisions': True,
        'drag_coefficients': [
            [
                [-1e-1, -1e1, -1e1]
                if i < 12
                else [-1e0, -1e0, -1e0]  # [-1e-1, -1e-1, -1e-1]
                if (i - 12) % 4 > 1
                else [0, 0, 0],
                [-1e-8, -1e-8, -1e-8],
            ]
            for i in range(12+4*4)
        ],
        'drives_init': [2, 0],
        'weight_osc_body': 1e1,
        'weight_osc_legs_internal': 3e1,
        'weight_osc_legs_opposite': 1e0,  # 1e1,
        'weight_osc_legs_following': 0,  # 1e1,
        'weight_osc_legs2body': 3e1,
        'weight_sens_contact_intralimb': -0.5,
        'weight_sens_contact_opposite': 2,
        'weight_sens_contact_following': 0,
        'weight_sens_contact_diagonal': 0,
        'weight_sens_hydro_freq': -1e-1,
        'weight_sens_hydro_amp': -1e-1,
        'body_stand_amplitude': 0.2,
        'body_stand_shift': np.pi/2,
        'legs_amplitudes': [np.pi/4, np.pi/32, np.pi/4, np.pi/8],
        'legs_offsets_walking': [0, np.pi/32, 0, np.pi/8],
        'modular_phases': np.array([3*np.pi/2, 0, 3*np.pi/2, 0]) - np.pi/4,
        'modular_amplitudes': np.full(4, 1.0),
        'default_lateral_friction': 2,
        'muscle_alpha': 5e0,
        'muscle_beta': -3e0,
        'muscle_gamma': 3e0,
        'muscle_delta': -2e-3,
    }
    kwargs_options.update(kwargs)
    return kwargs_options


def get_salamander_options(**kwargs):
    """Salamander options"""
    kwargs_options = get_salamander_kwargs_options(**kwargs)
    return get_animat_options(**kwargs_options)


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
        spawn_loader=SpawnLoader.FARMS,  # SpawnLoader.PYBULLET,
        default_control_type=ControlType.POSITION,
        swimming=False,
        n_legs=4,
        n_dof_legs=4,
        n_joints_body=13,
        use_self_collisions=True,
        body_stand_amplitude=0.2,
        legs_amplitudes=[np.pi/8, np.pi/16, np.pi/8, np.pi/8],
        legs_offsets_walking=[0, -np.pi/32, -np.pi/16, 0],
        legs_offsets_swimming=[2*np.pi/5, 0, 0, np.pi/2],
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
    links_names = kwargs.pop(
        'links_names',
        [
            # Body
            'base_link',
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
    feet = kwargs.pop(
        'feet',
        [
            'TS_FR',
            'TS_FL',
            'TS_HR',
            'TS_HL',
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
        gain_amplitude = [0]*(5+4*4)  # np.ones(5+4*4)
        # gain_amplitude[5] = 0
        for leg_i in range(2):
            for side_i in range(2):
                mirror = (1 if side_i else -1)
                mirror2 = (-1 if leg_i else 1)
                gain_amplitude[5+2*leg_i*4+side_i*4+0] = 0.5*mirror
                gain_amplitude[5+2*leg_i*4+side_i*4+1] = 3*mirror  # mirror
                gain_amplitude[5+2*leg_i*4+side_i*4+2] = mirror2  # -mirror
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
                joints_offsets[5+2*leg_i*4+side_i*4+2] = 0.5*np.pi*mirror2  # (
                #     0 if leg_i else -mirror*np.pi/3
                # )
                joints_offsets[5+2*leg_i*4+side_i*4+3] = 0  # (
                #     -mirror*np.pi/4 if leg_i else mirror*np.pi/16
                # )
        joints_offsets = dict(zip(joints_names, joints_offsets))

    # Animat options
    kwargs_options = dict(
        spawn_loader=SpawnLoader.PYBULLET,  # SpawnLoader.FARMS,
        spawn_position=[0, 0, 0.5],
        spawn_orientation=[0, 0, 0],
        default_control_type=ControlType.POSITION,
        swimming=False,
        n_legs=4,
        n_dof_legs=4,
        n_joints_body=5,
        use_self_collisions=True,
        body_stand_amplitude=0.2,
        legs_amplitudes=[np.pi/8, np.pi/16, np.pi/8, np.pi/8],
        legs_offsets_walking=[0, -np.pi/32, -np.pi/16, 0],
        legs_offsets_swimming=[2*np.pi/5, 0, 0, np.pi/2],
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
            # Body
            'base_link',
            'OROBOT',
            'PECTORAL_GIRDLE2',
            'S_SPINE3_C',
            'S_SPINE4_C',
            'S_SPINE5_C',
            'HIND_GIRDLE_C',
            'SPINE6_BACK_TT_C',
            'S_SPINE8_C',
            # Limb (FL)
            'LEFT_FRONT_YAW_C',
            'LEFT_FRONT_PITCH_C',
            'LEFT_FRONT_ELBOW_C',
            'LEFT_ULNA_T_C',
            # Limb (FR)
            'RIGHT_FRONT_YAW_C',
            'RIGHT_FRONT_PITCH_C',
            'RIGHT_FRONT_ELBOW_C',
            'RIGHT_ULNA_T_C',
            # Limb (HL)
            'LEFT_HIP_PITCH_C',
            'LEFT_HIND_ROLL_T_C',
            'LEFT_KNEE_T_C',
            'LEFT_CRUS_T_C',
            # Limb (HR)
            'RIGHT_HIP_PITCH_C',
            'RIGHT_HIND_ROLL_T_C',
            'RIGHT_KNEE_T_C',
            'RIGHT_CRUS_T_C',
        ]
    )
    joints_names = kwargs.pop('joints_names', [
        # Body
        'S_SPINE1_C_to_S_SPINE1T_S_SPINE1',
        'S_SPINE2_C_to_S_SPINE2T_S_SPINE2',
        'S_SPINE3_C_to_S_SPINE3T_S_SPINE3',
        'S_SPINE4_C_to_S_SPINE4T_S_SPINE4',
        'S_SPINE5_C_to_S_SPINE5T_S_SPINE5',
        'S_SPINE6_C_to_S_SPINE6T_S_SPINE6',
        'S_SPINE7_C_to_S_SPINE7T_S_SPINE7',
        'S_SPINE8_C_to_S_SPINE8T_S_SPINE8',
        # Limb (FL)
        'LEFT_FRONT_YAW_C_to_LEFT_FRONT_YAWT_LEFT_FRONT_YAW',
        'LEFT_FRONT_PITCH_C_to_LEFT_FRONT_PITCHT_LEFT_FRONT_PITCH',
        'LEFT_FRONT_WRIST_C_to_LEFT_FRONT_WRISTT_LEFT_FRONT_WRIST',
        'LEFT_FRONT_ELBOW_C_to_LEFT_FRONT_ELBOWT_LEFT_FRONT_ELBOW',
        # Limb (FR)
        'RIGHT_FRONT_YAW_C_to_RIGHT_FRONT_YAWT_RIGHT_FRONT_YAW',
        'RIGHT_FRONT_PITCH_C_to_RIGHT_FRONT_PITCHT_RIGHT_FRONT_PITCH',
        'RIGHT_FRONT_WRIST_C_to_RIGHT_FRONT_WRISTT_RIGHT_FRONT_WRIST',
        'RIGHT_FRONT_ELBOW_C_to_RIGHT_FRONT_ELBOWT_RIGHT_FRONT_ELBOW',
        # Limb (HL)
        'LEFT_HIND_YAW_C_to_LEFT_HIND_YAWT_LEFT_HIND_YAW',
        'LEFT_HIND_PITCH_C_to_LEFT_HIND_PITCHT_LEFT_HIND_PITCH',
        'LEFT_HIND_WRIST_C_to_LEFT_HIND_WRISTT_LEFT_HIND_WRIST',
        'LEFT_HIND_KNEE_C_to_LEFT_HIND_KNEET_LEFT_HIND_KNEE',
        # Limb (HR)
        'RIGHT_HIND_YAW_C_to_RIGHT_HIND_YAWT_RIGHT_HIND_YAW',
        'RIGHT_HIND_PITCH_C_to_RIGHT_HIND_PITCHT_RIGHT_HIND_PITCH',
        'RIGHT_HIND_WRIST_C_to_RIGHT_HIND_WRISTT_RIGHT_HIND_WRIST',
        'RIGHT_HIND_KNEE_C_to_RIGHT_HIND_KNEET_RIGHT_HIND_KNEE',
    ])
    feet = kwargs.pop(
        'feet',
        [
            'LEFT_ULNA_T_C',
            'RIGHT_ULNA_T_C',
            'LEFT_CRUS_T_C',
            'RIGHT_CRUS_T_C',
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
        gain_amplitude = [0]*(8+4*4)  # np.ones(8+4*4)
        # gain_amplitude[8] = 0
        for leg_i in range(2):
            for side_i in range(2):
                mirror = (1 if side_i else -1)
                mirror2 = (-1 if leg_i else 1)
                gain_amplitude[8+2*leg_i*4+side_i*4+0] = 0  # 0.5*mirror
                gain_amplitude[8+2*leg_i*4+side_i*4+1] = 0  # 3*mirror  # mirror
                gain_amplitude[8+2*leg_i*4+side_i*4+2] = 0  # mirror2  # -mirror
                gain_amplitude[8+2*leg_i*4+side_i*4+3] = 0  # mirror
        gain_amplitude = dict(zip(joints_names, gain_amplitude))

    # Joints joints_offsets
    if joints_offsets is None:
        joints_offsets = [0]*(8+4*4)
        for leg_i in range(2):
            for side_i in range(2):
                mirror = (1 if side_i else -1)
                mirror2 = (-1 if leg_i else 1)
                joints_offsets[8+2*leg_i*4+side_i*4+0] = 0  # -0.1*np.pi*mirror
                joints_offsets[8+2*leg_i*4+side_i*4+1] = 0  # 0*np.pi*mirror
                joints_offsets[8+2*leg_i*4+side_i*4+2] = 0  # 0.5*np.pi*mirror2
                joints_offsets[8+2*leg_i*4+side_i*4+3] = 0
        joints_offsets = dict(zip(joints_names, joints_offsets))

    # Animat options
    kwargs_options = dict(
        spawn_loader=SpawnLoader.PYBULLET,  # SpawnLoader.FARMS,
        spawn_position=[0, 0, 0.5],
        spawn_orientation=[0, 0, 0],
        default_control_type=ControlType.POSITION,
        swimming=False,
        n_legs=4,
        n_dof_legs=4,
        n_joints_body=8,
        use_self_collisions=True,
        body_stand_amplitude=0.2,
        legs_amplitudes=[np.pi/8, np.pi/16, np.pi/8, np.pi/8],
        legs_offsets_walking=[0, -np.pi/32, -np.pi/16, 0],
        legs_offsets_swimming=[2*np.pi/5, 0, 0, np.pi/2],
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
