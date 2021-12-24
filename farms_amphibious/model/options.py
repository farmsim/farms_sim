"""Animat options"""

from functools import partial
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
from farms_data.options import Options
from farms_mujoco.model.control import ControlType
from farms_mujoco.model.options import (
    ModelOptions,
    MorphologyOptions,
    LinkOptions,
    JointOptions,
    SpawnOptions,
    ControlOptions,
    JointControlOptions,
    SensorsOptions,
)
from .convention import AmphibiousConvention


def options_kwargs_keys():
    """Options kwargs keys"""
    return [
        'overlap',
        'weight_osc_body_side',
        'weight_osc_body_down',
        'weight_osc_legs_internal',
        'weight_osc_legs_opposite',
        'weight_osc_legs_following',
        'weight_osc_legs2body',
        'weight_osc_body2legs',
        'weight_sens_stretch_freq_up',
        'weight_sens_stretch_freq_same',
        'weight_sens_stretch_freq_down',
        'weight_sens_stretch_amp_up',
        'weight_sens_stretch_amp_same',
        'weight_sens_stretch_amp_down',
        'weight_sens_contact_intralimb',
        'weight_sens_contact_opposite',
        'weight_sens_contact_following',
        'weight_sens_contact_diagonal',
        'weight_sens_hydro_freq_up',
        'weight_sens_hydro_freq_down',
        'weight_sens_hydro_amp_up',
        'weight_sens_hydro_amp_down',
    ]


class AmphibiousOptions(ModelOptions):
    """Simulation options"""

    def __init__(self, **kwargs):
        super().__init__(
            spawn=SpawnOptions(**kwargs.pop('spawn')),
            morphology=AmphibiousMorphologyOptions(**kwargs.pop('morphology')),
            control=AmphibiousControlOptions(**kwargs.pop('control')),
        )
        self.physics = AmphibiousPhysicsOptions(**kwargs.pop('physics'))
        self.show_hydrodynamics = kwargs.pop('show_hydrodynamics')
        self.scale_hydrodynamics = kwargs.pop('scale_hydrodynamics')
        self.mujoco = kwargs.pop('mujoco')
        assert not kwargs, 'Unknown kwargs: {}'.format(kwargs)

    @classmethod
    def default(cls):
        """Deafault options"""
        return cls.from_options({})

    @classmethod
    def from_options(cls, kwargs=None):
        """From options"""
        options = {}
        options['morphology'] = kwargs.pop(
            'morphology',
            AmphibiousMorphologyOptions.from_options(kwargs)
        )
        convention = AmphibiousConvention(**options['morphology'])
        options['spawn'] = kwargs.pop(
            'spawn',
            SpawnOptions.from_options(kwargs)
        )
        options['physics'] = kwargs.pop(
            'physics',
            AmphibiousPhysicsOptions.from_options(kwargs)
        )
        options['mujoco'] = kwargs.pop('mujoco', {})
        if 'control' in kwargs:
            options['control'] = kwargs.pop('control')
        else:
            options['control'] = AmphibiousControlOptions.from_options(kwargs)
            options['control'].defaults_from_convention(convention, kwargs)
        options['show_hydrodynamics'] = kwargs.pop('show_hydrodynamics', False)
        options['scale_hydrodynamics'] = kwargs.pop('scale_hydrodynamics', 1)
        assert not kwargs, 'Unknown kwargs: {}'.format(kwargs)
        return cls(**options)


class AmphibiousMorphologyOptions(MorphologyOptions):
    """Amphibious morphology options"""

    def __init__(self, **kwargs):
        super().__init__(
            links=[
                AmphibiousLinkOptions(**link)
                for link in kwargs.pop('links')
            ],
            self_collisions=kwargs.pop('self_collisions'),
            joints=[
                JointOptions(**joint)
                for joint in kwargs.pop('joints')
            ],
        )
        self.n_joints_body = kwargs.pop('n_joints_body')
        self.n_dof_legs = kwargs.pop('n_dof_legs')
        self.n_legs = kwargs.pop('n_legs')
        assert not kwargs, 'Unknown kwargs: {}'.format(kwargs)

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['n_joints_body'] = kwargs.pop('n_joints_body')
        options['n_dof_legs'] = kwargs.pop('n_dof_legs')
        options['n_legs'] = kwargs.pop('n_legs')
        links_names = kwargs.pop(
            'links_names',
            AmphibiousConvention(**options).links_names,
        )
        convention = AmphibiousConvention(links_names=links_names, **options)
        default_lateral_friction = kwargs.pop('default_lateral_friction', 1)
        feet_friction = kwargs.pop('feet_friction', None)
        if feet_friction is None:
            feet_friction = default_lateral_friction
        feet_links = kwargs.pop('feet_links', convention.feet_links_names())
        links_friction_lateral = kwargs.pop(
            'links_friction_lateral',
            [
                feet_friction
                if link in feet_links
                else default_lateral_friction
                for link in links_names
            ],
        )
        links_friction_spinning = kwargs.pop(
            'links_friction_spinning',
            [0 for link in links_names],
        )
        links_friction_rolling = kwargs.pop(
            'links_friction_rolling',
            [0 for link in links_names],
        )
        links_no_collisions = kwargs.pop('links_no_collisions', (
            [
                convention.bodylink2name(body_i)
                for body_i in range(1, options['n_joints_body'])
            ] + [
                convention.leglink2name(leg_i, side_i, joint_i)
                for leg_i in range(options['n_legs']//2)
                for side_i in range(2)
                for joint_i in range(options['n_dof_legs']-1)
            ] if kwargs.pop('reduced_collisions', False) else []
        ))
        links_linear_damping = kwargs.pop(
            'links_linear_damping',
            [0 for link in links_names],
        )
        links_angular_damping = kwargs.pop(
            'links_angular_damping',
            [0 for link in links_names],
        )
        default_restitution = kwargs.pop('default_restitution', 0)
        links_restitution = kwargs.pop(
            'links_restitution',
            [default_restitution for link in links_names],
        )
        links_density = kwargs.pop('density', None)
        links_swimming = kwargs.pop('links_swimming', links_names)
        links_mass_multiplier = kwargs.pop('mass_multiplier', 1)
        drag_coefficients = kwargs.pop(
            'drag_coefficients',
            [None for name in links_names],
        )
        assert (
            len(links_names)
            == len(links_friction_lateral)
            == len(links_friction_spinning)
            == len(links_friction_rolling)
            == len(drag_coefficients)
        ), (
            'links_name,'
            ' links_friction_lateral,'
            ' links_friction_spinning,'
            ' links_friction_rolling,'
            ' drag_coefficients',
            np.shape(links_names),
            np.shape(links_friction_lateral),
            np.shape(links_friction_spinning),
            np.shape(links_friction_rolling),
            np.shape(drag_coefficients),
            links_names,
        )
        options['links'] = kwargs.pop(
            'links',
            [
                AmphibiousLinkOptions(
                    name=name,
                    collisions=name not in links_no_collisions,
                    density=links_density,
                    mass_multiplier=links_mass_multiplier,
                    swimming=name in links_swimming,
                    drag_coefficients=drag,
                    pybullet_dynamics={
                        'linearDamping': linear,
                        'angularDamping': angular,
                        'restitution': restitution,
                        'lateralFriction': lateral,
                        'spinningFriction': spin,
                        'rollingFriction': roll,
                        # 'contactStiffness': 1e2,
                        # 'contactDamping': 1e-3,
                    },
                )
                for (
                        name, lateral, spin, roll,
                        drag, linear, angular, restitution
                ) in zip(
                    links_names,
                    links_friction_lateral,
                    links_friction_spinning,
                    links_friction_rolling,
                    drag_coefficients,
                    links_linear_damping,
                    links_angular_damping,
                    links_restitution,
                )
            ]
        )
        options['self_collisions'] = kwargs.pop('self_collisions', [])
        joints_names = kwargs.pop('joints_names', convention.joints_names)
        joints_positions = kwargs.pop(
            'joints_positions',
            [0 for name in joints_names]
        )
        joints_velocities = kwargs.pop(
            'joints_velocities',
            [0 for name in joints_names]
        )
        joints_damping = kwargs.pop(
            'joints_damping',
            [0 for name in joints_names]
        )
        max_torque = kwargs.pop('max_torque', np.inf)
        max_velocity = kwargs.pop('max_velocity', np.inf)
        options['joints'] = kwargs.pop(
            'joints',
            [
                JointOptions(
                    name=name,
                    initial_position=position,
                    initial_velocity=velocity,
                    pybullet_dynamics={
                        'jointDamping': damping,
                        'maxJointVelocity': max_velocity,
                        'jointLimitForce': max_torque,
                    },
                )
                for name, position, velocity, damping in zip(
                    joints_names,
                    joints_positions,
                    joints_velocities,
                    joints_damping,
                )
            ]
        )
        morphology = cls(**options)
        if kwargs.pop('use_self_collisions', False):
            convention = AmphibiousConvention(**morphology)
            morphology.self_collisions = [
                # Body-body collisions
                [
                    convention.bodylink2name(body0),
                    convention.bodylink2name(body1),
                ]
                for body0 in range(options['n_joints_body']+1)
                for body1 in range(options['n_joints_body']+1)
                if abs(body1 - body0) > 3  # Avoid neighbouring collisions
            ] + [
                # Body-leg collisions
                [
                    convention.bodylink2name(body0),
                    convention.leglink2name(leg_i, side_i, joint_i),
                ]
                for body0 in range(options['n_joints_body']+1)
                for leg_i in range(options['n_legs']//2)
                for side_i in range(2)
                for joint_i in [options['n_dof_legs']-1]  # End-effector
            ] + [
                # Leg-leg collisions
                [
                    convention.leglink2name(leg0, side0, joint0),
                    convention.leglink2name(leg1, side1, joint1),
                ]
                for leg0 in range(options['n_legs']//2)
                for leg1 in range(options['n_legs']//2)
                for side0 in range(2)
                for side1 in range(2)
                for joint0 in [options['n_dof_legs']-1]
                for joint1 in [options['n_dof_legs']-1]
                if leg0!=leg1 or side0!=side1 or joint0!=joint1
            ]
            for links in morphology.self_collisions:
                assert links[0]!=links[1], 'Collision to self: {}'.format(links)
        return morphology

    def n_joints_legs(self):
        """Number of legs joints"""
        return self.n_legs*self.n_dof_legs


class AmphibiousLinkOptions(LinkOptions):
    """Amphibious link options

    The Pybullet dynamics represent the input arguments called with
    pybullet.changeDynamics(...).
    """

    def __init__(self, **kwargs):
        super().__init__(
            name=kwargs.pop('name'),
            collisions=kwargs.pop('collisions'),
            mass_multiplier=kwargs.pop('mass_multiplier'),
            pybullet_dynamics=kwargs.pop('pybullet_dynamics', {}),
        )
        self.density = kwargs.pop('density')
        self.swimming = kwargs.pop('swimming')
        self.drag_coefficients = kwargs.pop('drag_coefficients')
        assert not kwargs, 'Unknown kwargs: {}'.format(kwargs)


class AmphibiousPhysicsOptions(Options):
    """Amphibious physics options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.drag = kwargs.pop('drag')
        self.sph = kwargs.pop('sph')
        self.buoyancy = kwargs.pop('buoyancy')
        self.viscosity = kwargs.pop('viscosity')
        self.water_height = kwargs.pop('water_height')
        self.water_density = kwargs.pop('water_density')
        self.water_velocity = kwargs.pop('water_velocity')
        self.water_maps = kwargs.pop('water_maps')
        assert not kwargs, 'Unknown kwargs: {}'.format(kwargs)

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['drag'] = kwargs.pop('drag', False)
        options['sph'] = kwargs.pop('sph', False)
        options['buoyancy'] = kwargs.pop(
            'buoyancy',
            options['drag'] and not options['sph']
        )
        options['water_height'] = kwargs.pop(
            'water_height',
            options['drag'] or options['sph']
        )
        options['viscosity'] = kwargs.pop('viscosity', 1.0)
        options['water_density'] = kwargs.pop('water_density', 1000.0)
        options['water_velocity'] = kwargs.pop('water_velocity', [0.0, 0.0, 0.0])
        options['water_maps'] = kwargs.pop('water_maps', [])
        return cls(**options)


class AmphibiousControlOptions(ControlOptions):
    """Amphibious control options"""

    def __init__(self, **kwargs):
        super().__init__(
            sensors=(AmphibiousSensorsOptions(**kwargs.pop('sensors'))),
            joints=[
                AmphibiousJointControlOptions(**joint)
                for joint in kwargs.pop('joints')
            ],
        )
        self.kinematics_file = kwargs.pop('kinematics_file')
        self.manta_controller = kwargs.pop('manta_controller', False)
        self.kinematics_sampling = kwargs.pop('kinematics_sampling')
        network = kwargs.pop('network')
        self.network = (
            AmphibiousNetworkOptions(**network)
            if network is not None
            else None
        )
        if not self.kinematics_file:
            self.muscles = [
                AmphibiousMuscleSetOptions(**muscle)
                for muscle in kwargs.pop('muscles')
            ]
        else:
            self.muscles = kwargs.pop('muscles', None)
        assert not kwargs, 'Unknown kwargs: {}'.format(kwargs)

    @classmethod
    def options_from_kwargs(cls, kwargs):
        """Options from kwargs"""
        options = super(cls, cls).options_from_kwargs({
            'sensors': kwargs.pop(
                'sensors',
                AmphibiousSensorsOptions.options_from_kwargs(kwargs),
            ),
            'joints': kwargs.pop('joints', {}),
        })
        options['kinematics_file'] = kwargs.pop('kinematics_file', '')
        options['kinematics_sampling'] = kwargs.pop('kinematics_sampling', 0)
        options['network'] = kwargs.pop(
            'network',
            AmphibiousNetworkOptions.from_options(kwargs).to_dict()
        )
        if not options['kinematics_file']:
            options['muscles'] = kwargs.pop('muscles', [])
        return cls(**options)

    def defaults_from_convention(self, convention, kwargs):
        """Defaults from convention"""
        self.sensors.defaults_from_convention(convention, kwargs)
        self.network.defaults_from_convention(convention, kwargs)

        # Joints
        n_joints = convention.n_joints()
        offsets = [None]*n_joints

        # Turning body
        for joint_i in range(convention.n_joints_body):
            for side_i in range(2):
                offsets[convention.bodyjoint2index(joint_i=joint_i)] = {
                    'gain': 1,
                    'bias': 0,
                    'low': 1,
                    'high': 5,
                    'saturation': 0,
                }

        # Turning legs
        legs_offsets_walking = kwargs.pop(
            'legs_offsets_walking',
            [0, np.pi/32, 0, np.pi/8]
        )
        legs_offsets_swimming = kwargs.pop(
            'legs_offsets_swimming',
            [-np.pi/3, 0, 0, 0]
        )
        leg_turn_gain = kwargs.pop(
            'leg_turn_gain',
            [-1, 1]
            if convention.n_legs == 4
            else (-np.ones(convention.n_legs//2)).tolist()
        )
        leg_side_turn_gain = kwargs.pop(
            'leg_side_turn_gain',
            [-1, 1]
        )
        leg_joint_turn_gain = kwargs.pop(
            'leg_joint_turn_gain',
            [1, -1, 0, -1, 0],
        )

        # Augment parameters
        repeat = partial(np.repeat, repeats=convention.n_legs//2, axis=0)
        if np.ndim(legs_offsets_walking) == 1:
            legs_offsets_walking = repeat([legs_offsets_walking]).tolist()
        if np.ndim(legs_offsets_swimming) == 1:
            legs_offsets_swimming = repeat([legs_offsets_swimming]).tolist()
        if np.ndim(leg_side_turn_gain) == 1:
            leg_side_turn_gain = repeat([leg_side_turn_gain]).tolist()
        if np.ndim(leg_joint_turn_gain) == 1:
            leg_joint_turn_gain = repeat([leg_joint_turn_gain]).tolist()

        # Joints offsets for walking and swimming
        for leg_i in range(convention.n_legs//2):
            for side_i in range(2):
                for joint_i in range(convention.n_dof_legs):
                    offsets[convention.legjoint2index(
                        leg_i=leg_i,
                        side_i=side_i,
                        joint_i=joint_i,
                    )] = {
                        'gain': (
                            leg_turn_gain[leg_i]
                            *leg_side_turn_gain[leg_i][side_i]
                            *leg_joint_turn_gain[leg_i][joint_i]
                        ),
                        'bias': legs_offsets_walking[leg_i][joint_i],
                        'low': 1,
                        'high': 3,
                        'saturation': legs_offsets_swimming[leg_i][joint_i],
                    }

        # Amphibious joints control
        if not self.joints:
            self.joints = [
                AmphibiousJointControlOptions(
                    joint_name=None,
                    control_types=[],
                    max_torque=None,
                    equation=None,
                    transform=AmphibiousJointControlTransformOptions(
                        gain=None,
                        bias=None,
                    ),
                    offsets=AmphibiousJointControlOffsetOptions(
                        gain=None,
                        bias=None,
                        low=None,
                        high=None,
                        saturation=None,
                        rate=None,
                    ),
                    passive=AmphibiousPassiveJointOptions(
                        is_passive=False,
                        stiffness_coefficient=0,
                        damping_coefficient=0,
                        friction_coefficient=0,
                    )
                )
                for joint in range(n_joints)
            ]
        joints_names = kwargs.pop(
            'joints_control_names',
            convention.joints_names,
        )
        joints_rates = kwargs.pop(
            'joints_rates',
            {joint_name: 2 for joint_name in joints_names},
        )
        transform_gain = kwargs.pop(
            'transform_gain',
            {joint_name: 1 for joint_name in joints_names},
        )
        transform_bias = kwargs.pop(
            'transform_bias',
            {joint_name: 0 for joint_name in joints_names},
        )
        default_max_torque = kwargs.pop('default_max_torque', np.inf)
        max_torques = kwargs.pop(
            'max_torques',
            {joint_name: default_max_torque for joint_name in joints_names},
        )
        default_equation = kwargs.pop('default_equation', 'position')
        equations = kwargs.pop(
            'equations',
            {joint_name: default_equation for joint_name in joints_names},
        )
        for joint_i, joint in enumerate(self.joints):

            # Control
            if joint.joint_name is None:
                joint.joint_name = joints_names[joint_i]
            if joint.equation is None:
                joint.equation = equations[joint.joint_name]
            if not joint.control_types:
                joint.control_types = {
                    'position': [ControlType.POSITION],
                    'ekeberg_muscle': [ControlType.VELOCITY, ControlType.TORQUE],
                    'ekeberg_muscle_explicit': [ControlType.TORQUE],
                    'passive': [ControlType.VELOCITY, ControlType.TORQUE],
                    'passive_explicit': [ControlType.TORQUE],
                }[joint.equation]
            if joint.max_torque is None:
                joint.max_torque = max_torques[joint.joint_name]

            # Transform
            if joint.transform.gain is None:
                joint.transform.gain = transform_gain[joint.joint_name]
            if joint.transform.bias is None:
                joint.transform.bias = transform_bias[joint.joint_name]

            # Offset
            if joint.offsets.gain is None:
                joint.offsets.gain = offsets[joint_i]['gain']
            if joint.offsets.bias is None:
                joint.offsets.bias = offsets[joint_i]['bias']
            if joint.offsets.low is None:
                joint.offsets.low = offsets[joint_i]['low']
            if joint.offsets.high is None:
                joint.offsets.high = offsets[joint_i]['high']
            if joint.offsets.saturation is None:
                joint.offsets.saturation = offsets[joint_i]['saturation']
            if joint.offsets.rate is None:
                joint.offsets.rate = joints_rates[joint.joint_name]

        # Passive
        joints_passive = kwargs.pop('joints_passive', [])
        self.sensors.joints += [name for name, *_ in joints_passive]
        self.joints += [
            AmphibiousJointControlOptions(
                joint_name=joint_name,
                control_types=[ControlType.VELOCITY, ControlType.TORQUE],
                max_torque=np.inf,
                equation='passive',
                transform=AmphibiousJointControlTransformOptions(
                    gain=1,
                    bias=0,
                ),
                offsets=None,
                passive=AmphibiousPassiveJointOptions(
                    is_passive=True,
                    stiffness_coefficient=stiffness,
                    damping_coefficient=damping,
                    friction_coefficient=friction,
                ),
            )
            for joint_name, stiffness, damping, friction in joints_passive
        ]

        # Muscles
        if not self.kinematics_file:

            # Muscles
            if not self.muscles:
                self.muscles = [
                    AmphibiousMuscleSetOptions(
                        joint_name=None,
                        osc1=None,
                        osc2=None,
                        alpha=None,
                        beta=None,
                        gamma=None,
                        delta=None,
                        epsilon=None,
                    )
                    for muscle in range(n_joints)
                ]
            default_alpha = kwargs.pop('muscle_alpha', 0)
            default_beta = kwargs.pop('muscle_beta', 0)
            default_gamma = kwargs.pop('muscle_gamma', 0)
            default_delta = kwargs.pop('muscle_delta', 0)
            default_epsilon = kwargs.pop('muscle_epsilon', 0)
            for muscle_i, muscle in enumerate(self.muscles):
                if muscle.joint_name is None:
                    muscle.joint_name = joints_names[muscle_i]
                if muscle.osc1 is None:
                    muscle.osc1 = self.network.oscillators[2*muscle_i].name
                if muscle.osc2 is None:
                    muscle.osc2 = self.network.oscillators[2*muscle_i+1].name
                if muscle.alpha is None:
                    muscle.alpha = default_alpha
                if muscle.beta is None:
                    muscle.beta = default_beta
                if muscle.gamma is None:
                    muscle.gamma = default_gamma
                if muscle.delta is None:
                    muscle.delta = default_delta
                if muscle.epsilon is None:
                    muscle.epsilon = default_epsilon

    def joints_names(self):
        """Joints names"""
        return [joint.joint_name for joint in self.joints]

    def joints_offsets(self):
        """Joints offsets"""
        return [
            {
                'gain': joint.offsets.gain,
                'bias': joint.offsets.bias,
                'low': joint.offsets.low,
                'high': joint.offsets.high,
                'saturation': joint.offsets.saturation,
            }
            for joint in self.joints
            if joint.offsets is not None
        ]

    def joints_offset_rates(self):
        """Joints rates"""
        return [
            joint.offsets.rate
            for joint in self.joints
            if joint.offsets is not None
        ]

    def joints_transform_gain(self):
        """Joints gain amplitudes"""
        return [joint.transform.gain for joint in self.joints]

    def joints_transform_bias(self):
        """Joints offset bias"""
        return [joint.transform.bias for joint in self.joints]


class AmphibiousJointControlOptions(JointControlOptions):
    """Amphibious joint options"""

    def __init__(self, **kwargs):
        super().__init__(
            joint_name=kwargs.pop('joint_name'),
            control_types=kwargs.pop('control_types'),
            max_torque=kwargs.pop('max_torque'),
        )
        self.equation = kwargs.pop('equation')
        self.transform = kwargs.pop('transform')
        self.offsets = kwargs.pop('offsets')
        self.passive = kwargs.pop('passive')
        assert not kwargs, 'Unknown kwargs: {}'.format(kwargs)


class AmphibiousJointControlTransformOptions(Options):
    """Amphibious joint options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.gain = kwargs.pop('gain')
        self.bias = kwargs.pop('bias')
        assert not kwargs, 'Unknown kwargs: {}'.format(kwargs)


class AmphibiousJointControlOffsetOptions(Options):
    """Amphibious joint options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.gain = kwargs.pop('gain')
        self.bias = kwargs.pop('bias')
        self.low = kwargs.pop('low')
        self.high = kwargs.pop('high')
        self.saturation = kwargs.pop('saturation')
        self.rate = kwargs.pop('rate')
        assert not kwargs, 'Unknown kwargs: {}'.format(kwargs)


class AmphibiousSensorsOptions(SensorsOptions):
    """Amphibious sensors options"""

    def __init__(self, **kwargs):
        super().__init__(
            links=kwargs.pop('links'),
            joints=kwargs.pop('joints'),
            contacts=kwargs.pop('contacts'),
        )
        self.hydrodynamics = kwargs.pop('hydrodynamics')
        assert not kwargs, 'Unknown kwargs: {}'.format(kwargs)

    @classmethod
    def options_from_kwargs(cls, kwargs):
        """Options from kwargs"""
        options = super(cls, cls).options_from_kwargs(kwargs)
        options['hydrodynamics'] = kwargs.pop('sens_hydrodynamics', None)
        return options

    def defaults_from_convention(self, convention, kwargs):
        """Defaults from convention"""
        self.links = kwargs.pop(
            'sensors_links',
            convention.links_names
        )
        self.joints = kwargs.pop(
            'sensors_joints',
            convention.joints_names
        )
        self.contacts = kwargs.pop(
            'sensors_contacts',
            convention.feet_links_names()
        )
        self.hydrodynamics = kwargs.pop(
            'sensors_hydrodynamics',
            convention.links_names
        )


class AmphibiousNetworkOptions(Options):
    """Amphibious network options"""

    def __init__(self, **kwargs):
        super().__init__()

        # Drives
        self.drives = [
            AmphibiousDriveOptions(**drive)
            for drive in kwargs.pop('drives')
        ]

        # Oscillators
        self.oscillators = [
            AmphibiousOscillatorOptions(**oscillator)
            for oscillator in kwargs.pop('oscillators')
        ]

        # Connections
        self.osc2osc = kwargs.pop('osc2osc', None)
        self.drive2osc = kwargs.pop('drive2osc', None)
        self.joint2osc = kwargs.pop('joint2osc', None)
        self.contact2osc = kwargs.pop('contact2osc', None)
        self.hydro2osc = kwargs.pop('hydro2osc', None)

        # Kwargs
        assert not kwargs, 'Unknown kwargs: {}'.format(kwargs)

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['drives'] = kwargs.pop('drives', [])
        options['oscillators'] = kwargs.pop('oscillators', [])
        # Connectivity
        for option in [
                'osc2osc',
                'drive2osc',
                'joint2osc',
                'contact2osc',
                'hydro2osc',
        ]:
            options[option] = kwargs.pop(option, None)
        return cls(**options)

    def defaults_from_convention(self, convention, kwargs):
        """Defaults from convention"""

        # Parameters
        legs_amplitudes = kwargs.pop(
            'legs_amplitudes',
            [np.pi/4, np.pi/32, np.pi/4, np.pi/8]
        )

        # Augment parameters
        repeat = partial(np.repeat, repeats=convention.n_legs//2, axis=0)
        if np.ndim(legs_amplitudes) == 1:
            legs_amplitudes = repeat([legs_amplitudes]).tolist()

        # Drives
        if not self.drives:
            self.drives = [
                AmphibiousDriveOptions(
                    name=None,
                    initial_value=None,
                )
                for drive_i in range(2)
            ]
        drives_init = kwargs.pop('drives_init', [2, 0])
        for drive_i, drive in enumerate(self.drives):
            if drive.name is None:
                drive.name = 'Drive_{}'.format(drive_i)
            if drive.initial_value is None:
                drive.initial_value = drives_init[drive_i]

        # Oscillators
        n_oscillators = 2*convention.n_joints()
        if not self.oscillators:
            self.oscillators = [
                AmphibiousOscillatorOptions(
                    name=None,
                    initial_phase=None,
                    initial_amplitude=None,
                    input_drive=None,
                    frequency_gain=None,
                    frequency_bias=None,
                    frequency_low=None,
                    frequency_high=None,
                    frequency_saturation=None,
                    amplitude_gain=None,
                    amplitude_bias=None,
                    amplitude_low=None,
                    amplitude_high=None,
                    amplitude_saturation=None,
                    rate=None,
                    modular_phase=None,
                    modular_amplitude=None,
                )
                for osc_i in range(n_oscillators)
            ]
        state_init_smart = kwargs.pop('state_init_smart', False)
        random_state = RandomState(MT19937(SeedSequence(123456789)))
        state_init = kwargs.pop(
            'state_init',
            self.default_state_init(convention).tolist()
            if state_init_smart
            # else 1e-1*np.pi*np.linspace(0, 1, 5*convention.n_joints()),
            else 1e-1*random_state.rand(5*convention.n_joints()),
        )
        osc_frequencies = kwargs.pop(
            'osc_frequencies',
            self.default_osc_frequencies(convention, kwargs),
        )
        osc_amplitudes = kwargs.pop(
            'osc_amplitudes',
            self.default_osc_amplitudes(
                convention,
                body_walk_amplitude=kwargs.pop('body_walk_amplitude', 0.3),
                body_osc_gain=kwargs.pop('body_osc_gain', 0.25),
                body_osc_bias=kwargs.pop('body_osc_bias', 0.5),
                legs_amplitudes=legs_amplitudes,
            )
        )
        osc_rates = kwargs.pop(
            'osc_rates',
            self.default_osc_rates(convention),
        )
        osc_modular_phases = kwargs.pop(
            'osc_modular_phases',
            self.default_osc_modular_phases(
                convention=convention,
                phases=kwargs.pop('modular_phases', np.zeros(5)),
            )
        )
        osc_modular_amplitudes = kwargs.pop(
            'osc_modular_amplitudes',
            self.default_osc_modular_amplitudes(
                convention=convention,
                amplitudes=kwargs.pop('modular_amplitudes', np.zeros(5)),
            )
        )
        for osc_i, osc in enumerate(self.oscillators):
            if osc.name is None:
                osc.name = convention.oscindex2name(osc_i)
            if osc.initial_phase is None:
                osc.initial_phase = state_init[osc_i]
            if osc.initial_amplitude is None:
                osc.initial_amplitude = state_init[osc_i+n_oscillators]
            if osc.input_drive is None:
                osc.input_drive = 0
            if osc.frequency_gain is None:
                osc.frequency_gain = osc_frequencies[osc_i]['gain']
            if osc.frequency_bias is None:
                osc.frequency_bias = osc_frequencies[osc_i]['bias']
            if osc.frequency_low is None:
                osc.frequency_low = osc_frequencies[osc_i]['low']
            if osc.frequency_high is None:
                osc.frequency_high = osc_frequencies[osc_i]['high']
            if osc.frequency_saturation is None:
                osc.frequency_saturation = osc_frequencies[osc_i]['saturation']
            if osc.amplitude_gain is None:
                osc.amplitude_gain = osc_amplitudes[osc_i]['gain']
            if osc.amplitude_bias is None:
                osc.amplitude_bias = osc_amplitudes[osc_i]['bias']
            if osc.amplitude_low is None:
                osc.amplitude_low = osc_amplitudes[osc_i]['low']
            if osc.amplitude_high is None:
                osc.amplitude_high = osc_amplitudes[osc_i]['high']
            if osc.amplitude_saturation is None:
                osc.amplitude_saturation = osc_amplitudes[osc_i]['saturation']
            if osc.rate is None:
                osc.rate = osc_rates[osc_i]
            if osc.modular_phase is None:
                osc.modular_phase = osc_modular_phases[osc_i]
            if osc.modular_amplitude is None:
                osc.modular_amplitude = osc_modular_amplitudes[osc_i]

        # Connectivity
        overlap = kwargs.pop('overlap', True)
        if self.osc2osc is None:
            pi2 = 0.5*np.pi
            # body_stand_shift = kwargs.pop('body_stand_shift', pi2)
            n_leg_pairs = convention.n_legs//2
            legs_splits = np.array_split(
                np.arange(convention.n_joints_body),
                n_leg_pairs
            ) if convention.n_legs and convention.n_joints_body else None
            empty = np.array([], dtype=int)
            default_legbodyconnections = [  # Body osc to connect to each limb
                np.concatenate([
                    [split[0]-1]
                    if split_i
                    and len(split) <= len(legs_splits[split_i-1])
                    else empty,
                    split,
                    [split[-1]+1]
                    if split_i < (n_leg_pairs-1)
                    and len(split) <= len(legs_splits[split_i+1])
                    else empty,
                ])
                if overlap
                else split
                for split_i, split in enumerate(legs_splits)
            ] if legs_splits is not None else []
            standing = kwargs.pop('standing_wave', True)
            repeat = partial(np.repeat, repeats=convention.n_legs//2, axis=0)
            intralimb_phases = kwargs.pop(
                'intralimb_phases',
                [0, pi2, 0, pi2, 0],
            )
            if np.ndim(intralimb_phases) == 1:
                intralimb_phases = repeat([intralimb_phases]).tolist()
            self.osc2osc = (
                self.default_osc2osc(
                    convention=convention,
                    weight_body2body_down=kwargs.pop(
                        'weight_osc_body_down',
                        0,
                    ),
                    weight_body2body_side=kwargs.pop(
                        'weight_osc_body_side',
                        0,
                    ),
                    phase_body2body=kwargs.pop(
                        'body_phase_bias',
                        2*np.pi/convention.n_joints_body
                        if convention.n_joints_body > 0
                        else 0
                    ),
                    weight_intralimb=kwargs.pop(
                        'weight_osc_legs_internal',
                        0,
                    ),
                    weight_interlimb_opposite=kwargs.pop(
                        'weight_osc_legs_opposite',
                        0,
                    ),
                    weight_interlimb_following=kwargs.pop(
                        'weight_osc_legs_following',
                        0,
                    ),
                    weight_limb2body=kwargs.pop(
                        'weight_osc_legs2body',
                        0,
                    ),
                    weight_body2limb=kwargs.pop(
                        'weight_osc_body2legs',
                        0,
                    ),
                    intralimb_phases=intralimb_phases,
                    phase_limb_follow=kwargs.pop(
                        'leg_phase_follow',
                        np.pi,
                    ),
                    body_walk_phases=kwargs.pop(
                        'body_walk_phases',
                        # [
                        #     body_i*2*np.pi/convention.n_joints_body
                        #     + body_stand_shift
                        #     for body_i in range(convention.n_joints_body)
                        # ]
                        np.concatenate([
                            np.full(len(split), (np.pi*(split_i+1)) % (2*np.pi))
                            for split_i, split in enumerate(legs_splits)
                        ]).tolist()
                        if standing
                        and legs_splits is not None
                        else np.zeros(convention.n_joints_body)
                        if legs_splits is not None
                        else [],
                    ),
                    legbodyjoints=kwargs.pop(
                        'legbodyjoints',
                        range(1)
                        if kwargs.pop('reduced_limb_body', True)
                        else range(convention.n_dof_legs-1),
                    ),
                    legbodyconnections=kwargs.pop(
                        'legbodyconnections',
                        [
                            range(convention.n_joints_body)
                            for leg_i in range(n_leg_pairs)
                        ]
                        if kwargs.pop('full_leg_body', False)
                        else default_legbodyconnections,
                    ),
                    standing=standing,
                )
            )
        if self.joint2osc is None:
            self.joint2osc = self.default_joint2osc(
                convention,
                kwargs.pop('weight_sens_stretch_freq_up', 0),
                kwargs.pop('weight_sens_stretch_freq_same', 0),
                kwargs.pop('weight_sens_stretch_freq_down', 0),
                kwargs.pop('weight_sens_stretch_amp_up', 0),
                kwargs.pop('weight_sens_stretch_amp_same', 0),
                kwargs.pop('weight_sens_stretch_amp_down', 0),
            )
        if self.contact2osc is None:
            self.contact2osc = self.default_contact2osc(
                convention,
                kwargs.pop('weight_sens_contact_intralimb', 0),
                kwargs.pop('weight_sens_contact_opposite', 0),
                kwargs.pop('weight_sens_contact_following', 0),
                kwargs.pop('weight_sens_contact_diagonal', 0),
            )
        if self.hydro2osc is None:
            self.hydro2osc = self.default_hydro2osc(
                convention,
                kwargs.pop('weight_sens_hydro_freq_up', 0),
                kwargs.pop('weight_sens_hydro_freq_down', 0),
                kwargs.pop('weight_sens_hydro_amp_up', 0),
                kwargs.pop('weight_sens_hydro_amp_down', 0),
            )

    def drives_init(self):
        """Initial drives"""
        return [drive.initial_value for drive in self.drives]

    def state_init(self):
        """Initial states"""
        return [
            osc.initial_phase for osc in self.oscillators
        ] + [
            osc.initial_amplitude for osc in self.oscillators
        ] + [0 for osc in self.oscillators[::2]]

    def osc_names(self):
        """Oscillator names"""
        return [osc.name for osc in self.oscillators]

    def osc_frequencies(self):
        """Oscillator frequencies"""
        return [
            {
                'gain': osc.frequency_gain,
                'bias': osc.frequency_bias,
                'low': osc.frequency_low,
                'high': osc.frequency_high,
                'saturation': osc.frequency_saturation,
            }
            for osc in self.oscillators
        ]

    def osc_amplitudes(self):
        """Oscillator amplitudes"""
        return [
            {
                'gain': osc.amplitude_gain,
                'bias': osc.amplitude_bias,
                'low': osc.amplitude_low,
                'high': osc.amplitude_high,
                'saturation': osc.amplitude_saturation,
            }
            for osc in self.oscillators
        ]

    def osc_rates(self):
        """Oscillator rates"""
        return [osc.rate for osc in self.oscillators]

    def osc_modular_phases(self):
        """Oscillator modular phases"""
        return [osc.modular_phase for osc in self.oscillators]

    def osc_modular_amplitudes(self):
        """Oscillator modular amplitudes"""
        return [osc.modular_amplitude for osc in self.oscillators]

    @staticmethod
    def default_state_init(convention):
        """Default state"""
        state = np.zeros(5*convention.n_joints())
        phases_init_body = np.linspace(2*np.pi, 0, convention.n_joints_body)
        for joint_i in range(convention.n_joints_body):
            for side_osc in range(2):
                state[convention.bodyosc2index(
                    joint_i,
                    side=side_osc,
                )] = (
                    phases_init_body[joint_i]
                    + (0 if side_osc else np.pi)
                )
        phases_init_legs = [3*np.pi/2, 0, 3*np.pi/2, 0, 0]
        for joint_i in range(convention.n_dof_legs):
            for leg_i in range(convention.n_legs//2):
                for side_i in range(2):
                    for side in range(2):
                        state[convention.legosc2index(
                            leg_i,
                            side_i,
                            joint_i,
                            side=side,
                        )] = (
                            (0 if leg_i else np.pi)
                            + (0 if side_i else np.pi)
                            + (0 if side else np.pi)
                            + phases_init_legs[joint_i]
                        )
        state += 1e-3*np.arange(5*convention.n_joints())
        return state

    @staticmethod
    def default_osc_frequencies(convention, kwargs):
        """Walking parameters"""
        n_oscillators = 2*(convention.n_joints())
        frequencies = [None]*n_oscillators

        # Body
        body_freq_gain = kwargs.pop('body_freq_gain', 2*np.pi*0.55)
        body_freq_bias = kwargs.pop('body_freq_bias', 2*np.pi*0.0)
        for joint_i in range(convention.n_joints_body):
            for side in range(2):
                frequencies[convention.bodyosc2index(joint_i, side=side)] = {
                    # 'gain': 2*np.pi*0.2,
                    # 'bias': 2*np.pi*0.3,
                    # 'gain': 2*np.pi*0.55, # 1.65 - 2.75 [Hz]
                    # 'bias': 2*np.pi*0.0,
                    'gain': body_freq_gain,
                    'bias': body_freq_bias,
                    'low': 1,
                    'high': 5,
                    'saturation': 0,
                }

        # legs
        legs_freq_gain = kwargs.pop('legs_freq_gain', 2*np.pi*0.3)
        legs_freq_bias = kwargs.pop('legs_freq_bias', 2*np.pi*0.3)
        for joint_i in range(convention.n_dof_legs):
            for leg_i in range(convention.n_legs//2):
                for side_i in range(2):
                    for side in range(2):
                        frequencies[convention.legosc2index(
                            leg_i,
                            side_i,
                            joint_i,
                            side=side,
                        )] = {
                            # 'gain': 2*np.pi*0.2,
                            # 'bias': 2*np.pi*0.0,
                            # 'gain': 2*np.pi*0.3, # 0.6 - 1.2 [Hz]
                            # 'bias': 2*np.pi*0.3,
                            'gain': legs_freq_gain,
                            'bias': legs_freq_bias,
                            'low': 1,
                            'high': 3,
                            'saturation': 0,
                        }

        return frequencies

    @staticmethod
    def default_osc_amplitudes(
            convention,
            body_walk_amplitude,
            body_osc_gain,
            body_osc_bias,
            legs_amplitudes,
    ):
        """Walking parameters"""
        n_oscillators = 2*(convention.n_joints())
        amplitudes = [None]*n_oscillators
        # Body ampltidudes
        for joint_i in range(convention.n_joints_body):
            for side in range(2):
                amplitudes[convention.bodyosc2index(joint_i, side=side)] = {
                    'gain': body_osc_gain*body_walk_amplitude,
                    'bias': body_osc_bias*body_walk_amplitude,
                    'low': 1,
                    'high': 5,
                    'saturation': 0,
                }
        # Legs ampltidudes
        for leg_i in range(convention.n_legs//2):
            for joint_i in range(convention.n_dof_legs):
                amplitude = legs_amplitudes[leg_i][joint_i]
                for side_i in range(2):
                    for side in range(2):
                        amplitudes[convention.legosc2index(
                            leg_i,
                            side_i,
                            joint_i,
                            side=side,
                        )] = {
                            'gain': 0,
                            'bias': amplitude,
                            'low': 1,
                            'high': 3,
                            'saturation': 0,
                        }
        return amplitudes

    @staticmethod
    def default_osc_rates(convention):
        """Walking parameters"""
        n_oscillators = 2*(convention.n_joints())
        rates = 10*np.ones(n_oscillators)
        return rates.tolist()

    @staticmethod
    def default_osc_modular_phases(convention, phases):
        """Default"""
        n_oscillators = 2*(convention.n_joints())
        values = np.zeros(n_oscillators)
        for joint_i in range(convention.n_dof_legs):
            phase = phases[joint_i]
            for leg_i in range(convention.n_legs//2):
                for side_i in range(2):
                    for side in range(2):
                        values[convention.legosc2index(
                            leg_i,
                            side_i,
                            joint_i,
                            side=side,
                        )] = (
                            phase
                            # + (0 if leg_i else np.pi)
                            # + (0 if side_i else np.pi)
                            + (0 if side else np.pi)
                        )
        return values.tolist()

    @staticmethod
    def default_osc_modular_amplitudes(convention, amplitudes):
        """Default"""
        n_oscillators = 2*(convention.n_joints())
        values = np.zeros(n_oscillators)
        for joint_i in range(convention.n_dof_legs):
            amplitude = amplitudes[joint_i]
            for leg_i in range(convention.n_legs//2):
                for side_i in range(2):
                    for side in range(2):
                        values[convention.legosc2index(
                            leg_i,
                            side_i,
                            joint_i,
                            side=side,
                        )] = amplitude
        return values.tolist()

    @staticmethod
    def default_osc2osc(
            convention,
            weight_body2body_side,
            weight_body2body_down,
            phase_body2body,
            weight_intralimb,
            weight_interlimb_opposite,
            weight_interlimb_following,
            weight_limb2body,
            weight_body2limb,
            intralimb_phases,
            phase_limb_follow,
            body_walk_phases,
            legbodyjoints,
            legbodyconnections,
            standing,
    ):
        """Default oscillators to oscillators connectivity"""
        connectivity = []
        n_body_joints = convention.n_joints_body

        # Body
        if weight_body2body_side != 0:
            # Antagonist oscillators
            for i in range(n_body_joints):
                for sides in [[1, 0], [0, 1]]:
                    connectivity.append({
                        'in': convention.bodyosc2name(
                            joint_i=i,
                            side=sides[0]
                        ),
                        'out': convention.bodyosc2name(
                            joint_i=i,
                            side=sides[1]
                        ),
                        'type': 'OSC2OSC',
                        'weight': weight_body2body_side,
                        'phase_bias': np.pi,
                    })
        if weight_body2body_down != 0:
            # Following oscillators
            for i in range(n_body_joints-1):
                for side in range(2):
                    for osc, phase in [
                            [[i+1, i], +phase_body2body],
                            [[i, i+1], -phase_body2body]
                    ]:
                        connectivity.append({
                            'in': convention.bodyosc2name(
                                joint_i=osc[0],
                                side=side
                            ),
                            'out': convention.bodyosc2name(
                                joint_i=osc[1],
                                side=side
                            ),
                            'type': 'OSC2OSC',
                            'weight': weight_body2body_down,
                            'phase_bias': phase % (2*np.pi),
                        })

        # Legs (internal)
        if weight_intralimb != 0:
            for leg_i in range(convention.n_legs//2):
                for side_i in range(2):
                    _options = {
                        'leg_i': leg_i,
                        'side_i': side_i
                    }
                    # X - X
                    for joint_i in range(convention.n_dof_legs):
                        for sides in [[1, 0], [0, 1]]:
                            connectivity.append({
                                'in': convention.legosc2name(
                                    **_options,
                                    joint_i=joint_i,
                                    side=sides[0]
                                ),
                                'out': convention.legosc2name(
                                    **_options,
                                    joint_i=joint_i,
                                    side=sides[1]
                                ),
                                'type': 'OSC2OSC',
                                'weight': weight_intralimb,
                                'phase_bias': np.pi,
                            })

                    # Following
                    internal_connectivity = []
                    for joint_i_0 in range(convention.n_dof_legs):
                        for joint_i_1 in range(convention.n_dof_legs):
                            if joint_i_0 != joint_i_1:
                                phase = (
                                    intralimb_phases[leg_i][joint_i_1]
                                    - intralimb_phases[leg_i][joint_i_0]
                                )
                                internal_connectivity.extend([
                                    [[joint_i_1, joint_i_0], 0, phase],
                                    [[joint_i_1, joint_i_0], 1, phase],
                                ])
                    for joints, side, phase in internal_connectivity:
                        connectivity.append({
                            'in': convention.legosc2name(
                                **_options,
                                joint_i=joints[0],
                                side=side,
                            ),
                            'out': convention.legosc2name(
                                **_options,
                                joint_i=joints[1],
                                side=side,
                            ),
                            'type': 'OSC2OSC',
                            'weight': weight_intralimb,
                            'phase_bias': phase % (2*np.pi),
                        })

        # Opposite leg interaction
        if weight_interlimb_opposite != 0:
            for leg_i in range(convention.n_legs//2):
                for joint_i in range(convention.n_dof_legs):
                    for side in range(2):
                        _options = {
                            'joint_i': joint_i,
                            'side': side
                        }
                        for sides in [[1, 0], [0, 1]]:
                            connectivity.append({
                                'in': convention.legosc2name(
                                    leg_i=leg_i,
                                    side_i=sides[0],
                                    **_options
                                ),
                                'out': convention.legosc2name(
                                    leg_i=leg_i,
                                    side_i=sides[1],
                                    **_options
                                ),
                                'type': 'OSC2OSC',
                                'weight': weight_interlimb_opposite,
                                'phase_bias': np.pi,
                            })

        # Following leg interaction
        if weight_interlimb_following != 0:
            for leg_pre in range(convention.n_legs//2-1):
                for side_i in range(2):
                    for side in range(2):
                        _options = {
                            'side_i': side_i,
                            'side': side,
                            'joint_i': 0,
                        }
                        for legs, phase in [
                                [[leg_pre, leg_pre+1], phase_limb_follow],
                                [[leg_pre+1, leg_pre], -phase_limb_follow],
                        ]:
                            connectivity.append({
                                'in': convention.legosc2name(
                                    leg_i=legs[0],
                                    **_options
                                ),
                                'out': convention.legosc2name(
                                    leg_i=legs[1],
                                    **_options
                                ),
                                'type': 'OSC2OSC',
                                'weight': weight_interlimb_following,
                                'phase_bias': phase % (2*np.pi),
                            })

        # Legs-body interaction
        if weight_limb2body != 0:
            for leg_i in range(convention.n_legs//2):
                for side_i in range(2):
                    for joint_i in legbodyjoints:
                        for side_leg_osc in range(2):
                            for lateral in range(1):
                                for body_i in legbodyconnections[leg_i]:
                                    connectivity.append({
                                        'in': convention.bodyosc2name(
                                            joint_i=body_i,
                                            side=(side_i+lateral)%2
                                        ),
                                        'out': convention.legosc2name(
                                            leg_i=leg_i,
                                            side_i=side_i,
                                            joint_i=joint_i,
                                            side=side_leg_osc
                                        ),
                                        'type': 'OSC2OSC',
                                        'weight': weight_limb2body,
                                        'phase_bias': (
                                            body_walk_phases[body_i]
                                            + np.pi*(
                                                1
                                                + lateral
                                                + leg_i
                                                + side_leg_osc
                                            )
                                            - intralimb_phases[leg_i][joint_i]
                                        ) % (2*np.pi),
                                    })

        # Body-legs interaction
        if weight_body2limb != 0:
            for leg_i in range(convention.n_legs//2):
                for side_i in range(2):
                    for joint_i in legbodyjoints:
                        for side_leg_osc in range(2):
                            for lateral in range(1):
                                for body_i in legbodyconnections[leg_i]:
                                    connectivity.append({
                                        'in': convention.legosc2name(
                                            leg_i=leg_i,
                                            side_i=side_i,
                                            joint_i=joint_i,
                                            side=side_leg_osc
                                        ),
                                        'out': convention.bodyosc2name(
                                            joint_i=body_i,
                                            side=(side_i+lateral)%2
                                        ),
                                        'type': 'OSC2OSC',
                                        'weight': weight_body2limb,
                                        'phase_bias': (-(
                                            body_walk_phases[body_i]
                                            + np.pi*(
                                                1
                                                + lateral
                                                + (leg_i if standing else 0.5)
                                                + side_leg_osc
                                            )
                                            - intralimb_phases[leg_i][joint_i]
                                        )) % (2*np.pi),
                                    })
        return connectivity

    @staticmethod
    def default_joint2osc(
            convention,
            weight_frequency_up,
            weight_frequency_same,
            weight_frequency_down,
            weight_amplitude_up,
            weight_amplitude_same,
            weight_amplitude_down,
    ):
        """Default joint sensors to oscillators connectivity"""
        connectivity = []
        if weight_frequency_up:
            for joint_i in range(convention.n_joints_body-1):
                for side_osc in range(2):
                    connectivity.append({
                        'in': convention.bodyosc2name(
                            joint_i=joint_i,
                            side=side_osc,
                        ),
                        'out': convention.bodyjoint2name(joint_i+1),
                        'type': 'STRETCH2FREQ',
                        'weight': (1 if side_osc else 1)*weight_frequency_up,
                    })
        if weight_frequency_same:
            for joint_i in range(convention.n_joints_body):
                for side_osc in range(2):
                    connectivity.append({
                        'in': convention.bodyosc2name(
                            joint_i=joint_i,
                            side=side_osc,
                        ),
                        'out': convention.bodyjoint2name(joint_i),
                        'type': 'STRETCH2FREQ',
                        'weight': (1 if side_osc else 1)*weight_frequency_same,
                    })
        if weight_frequency_down:
            for joint_i in range(convention.n_joints_body-1):
                for side_osc in range(2):
                    connectivity.append({
                        'in': convention.bodyosc2name(
                            joint_i=joint_i+1,
                            side=side_osc,
                        ),
                        'out': convention.bodyjoint2name(joint_i),
                        'type': 'STRETCH2FREQ',
                        'weight': (1 if side_osc else 1)*weight_frequency_down,
                    })
        if weight_amplitude_up:
            for joint_i in range(convention.n_joints_body-1):
                for side_osc in range(2):
                    connectivity.append({
                        'in': convention.bodyosc2name(
                            joint_i=joint_i,
                            side=side_osc,
                        ),
                        'out': convention.bodyjoint2name(joint_i+1),
                        'type': 'STRETCH2AMP',
                        'weight': (1 if side_osc else 1)*weight_amplitude_up,
                    })
        if weight_amplitude_same:
            for joint_i in range(convention.n_joints_body):
                for side_osc in range(2):
                    connectivity.append({
                        'in': convention.bodyosc2name(
                            joint_i=joint_i,
                            side=side_osc,
                        ),
                        'out': convention.bodyjoint2name(joint_i),
                        'type': 'STRETCH2AMP',
                        'weight': (1 if side_osc else 1)*weight_amplitude_same,
                    })
        if weight_amplitude_down:
            for joint_i in range(convention.n_joints_body-1):
                for side_osc in range(2):
                    connectivity.append({
                        'in': convention.bodyosc2name(
                            joint_i=joint_i+1,
                            side=side_osc,
                        ),
                        'out': convention.bodyjoint2name(joint_i),
                        'type': 'STRETCH2AMP',
                        'weight': (1 if side_osc else 1)*weight_amplitude_down,
                    })
        return connectivity

    @staticmethod
    def default_contact2osc(
            convention,
            w_intralimb,
            w_opposite,
            w_following,
            w_diagonal
    ):
        """Default contact sensors to oscillators connectivity"""
        connectivity = []
        # Intralimb
        for sensor_leg_i in range(convention.n_legs//2):
            for sensor_side_i in range(2):
                for joint_i in range(convention.n_dof_legs):
                    for side_o in range(2):
                        if w_intralimb:
                            connectivity.append({
                                'in': convention.legosc2name(
                                    leg_i=sensor_leg_i,
                                    side_i=sensor_side_i,
                                    joint_i=joint_i,
                                    side=side_o
                                ),
                                'out': convention.contactleglink2name(
                                    leg_i=sensor_leg_i,
                                    side_i=sensor_side_i
                                ),
                                'type': 'REACTION2FREQ',
                                'weight': w_intralimb,
                            })
                        if w_opposite:
                            connectivity.append({
                                'in': convention.legosc2name(
                                    leg_i=sensor_leg_i,
                                    side_i=(sensor_side_i+1)%2,
                                    joint_i=joint_i,
                                    side=side_o
                                ),
                                'out': convention.contactleglink2name(
                                    leg_i=sensor_leg_i,
                                    side_i=sensor_side_i
                                ),
                                'type': 'REACTION2FREQ',
                                'weight': w_opposite,
                            })
                        if w_following:
                            if sensor_leg_i > 0:
                                connectivity.append({
                                    'in': convention.legosc2name(
                                        leg_i=sensor_leg_i-1,
                                        side_i=sensor_side_i,
                                        joint_i=joint_i,
                                        side=side_o
                                    ),
                                    'out': convention.contactleglink2name(
                                        leg_i=sensor_leg_i,
                                        side_i=sensor_side_i
                                    ),
                                    'type': 'REACTION2FREQ',
                                    'weight': w_following,
                                })
                            if sensor_leg_i < (convention.n_legs//2 - 1):
                                connectivity.append({
                                    'in': convention.legosc2name(
                                        leg_i=sensor_leg_i+1,
                                        side_i=sensor_side_i,
                                        joint_i=joint_i,
                                        side=side_o
                                    ),
                                    'out': convention.contactleglink2name(
                                        leg_i=sensor_leg_i,
                                        side_i=sensor_side_i
                                    ),
                                    'type': 'REACTION2FREQ',
                                    'weight': w_following,
                                })
                        if w_diagonal:
                            if sensor_leg_i > 0:
                                connectivity.append({
                                    'in': convention.legosc2name(
                                        leg_i=sensor_leg_i-1,
                                        side_i=(sensor_side_i+1)%2,
                                        joint_i=joint_i,
                                        side=side_o
                                    ),
                                    'out': convention.contactleglink2name(
                                        leg_i=sensor_leg_i,
                                        side_i=sensor_side_i
                                    ),
                                    'type': 'REACTION2FREQ',
                                    'weight': w_diagonal,
                                })
                            if sensor_leg_i < (convention.n_legs//2 - 1):
                                connectivity.append({
                                    'in': convention.legosc2name(
                                        leg_i=sensor_leg_i+1,
                                        side_i=(sensor_side_i+1)%2,
                                        joint_i=joint_i,
                                        side=side_o
                                    ),
                                    'out': convention.contactleglink2name(
                                        leg_i=sensor_leg_i,
                                        side_i=sensor_side_i
                                    ),
                                    'type': 'REACTION2FREQ',
                                    'weight': w_diagonal,
                                })
        return connectivity

    @staticmethod
    def default_hydro2osc(
            convention,
            weight_frequency_up,
            weight_frequency_down,
            weight_amplitude_up,
            weight_amplitude_down,
    ):
        """Default hydrodynamics sensors to oscillators connectivity"""
        connectivity = []
        for joint_i in range(convention.n_joints_body):
            for side_osc in range(2):
                if weight_frequency_up:
                    connectivity.append({
                        'in': convention.bodyosc2name(
                            joint_i=joint_i,
                            side=side_osc
                        ),
                        'out': convention.bodylink2name(joint_i+1),
                        'type': 'LATERAL2FREQ',
                        'weight': weight_frequency_up,
                    })
                if weight_frequency_down:
                    connectivity.append({
                        'in': convention.bodyosc2name(
                            joint_i=joint_i,
                            side=side_osc
                        ),
                        'out': convention.bodylink2name(joint_i),
                        'type': 'LATERAL2FREQ',
                        'weight': weight_frequency_down,
                    })
                if weight_amplitude_up:
                    connectivity.append({
                        'in': convention.bodyosc2name(
                            joint_i=joint_i,
                            side=side_osc
                        ),
                        'out': convention.bodylink2name(joint_i+1),
                        'type': 'LATERAL2AMP',
                        'weight': weight_amplitude_up,
                    })
                if weight_amplitude_down:
                    connectivity.append({
                        'in': convention.bodyosc2name(
                            joint_i=joint_i,
                            side=side_osc
                        ),
                        'out': convention.bodylink2name(joint_i),
                        'type': 'LATERAL2AMP',
                        'weight': weight_amplitude_down,
                    })
        return connectivity


class AmphibiousOscillatorOptions(Options):
    """Amphibious oscillator options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.name = kwargs.pop('name')
        self.initial_phase = kwargs.pop('initial_phase')
        self.initial_amplitude = kwargs.pop('initial_amplitude')
        self.input_drive = kwargs.pop('input_drive')
        self.frequency_gain = kwargs.pop('frequency_gain')
        self.frequency_bias = kwargs.pop('frequency_bias')
        self.frequency_low = kwargs.pop('frequency_low')
        self.frequency_high = kwargs.pop('frequency_high')
        self.frequency_saturation = kwargs.pop('frequency_saturation')
        self.amplitude_gain = kwargs.pop('amplitude_gain')
        self.amplitude_bias = kwargs.pop('amplitude_bias')
        self.amplitude_low = kwargs.pop('amplitude_low')
        self.amplitude_high = kwargs.pop('amplitude_high')
        self.amplitude_saturation = kwargs.pop('amplitude_saturation')
        self.rate = kwargs.pop('rate')
        self.modular_phase = kwargs.pop('modular_phase')
        self.modular_amplitude = kwargs.pop('modular_amplitude')
        assert not kwargs, 'Unknown kwargs: {}'.format(kwargs)


class AmphibiousDriveOptions(Options):
    """Amphibious drive options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.name: str = kwargs.pop('name')
        self.initial_value: float = kwargs.pop('initial_value')
        assert not kwargs, 'Unknown kwargs: {}'.format(kwargs)


class AmphibiousMuscleSetOptions(Options):
    """Amphibious muscle options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.joint_name: str = kwargs.pop('joint_name')
        self.osc1: str = kwargs.pop('osc1')
        self.osc2: str = kwargs.pop('osc2')
        self.alpha: float = kwargs.pop('alpha')  # Gain
        self.beta: float = kwargs.pop('beta')  # Stiffness gain
        self.gamma: float = kwargs.pop('gamma')  # Tonic gain
        self.delta: float = kwargs.pop('delta')  # Damping coefficient
        self.epsilon: float = kwargs.pop('epsilon')  # Friction coefficient
        assert not kwargs, 'Unknown kwargs: {}'.format(kwargs)


class AmphibiousPassiveJointOptions(Options):
    """Amphibious passive joint options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.is_passive: bool = kwargs.pop('is_passive')
        self.stiffness_coefficient: float = kwargs.pop('stiffness_coefficient')
        self.damping_coefficient: float = kwargs.pop('damping_coefficient')
        self.friction_coefficient: float = kwargs.pop('friction_coefficient')
        assert not kwargs, 'Unknown kwargs: {}'.format(kwargs)
