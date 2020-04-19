"""Animat options"""

import numpy as np

from farms_data.options import Options
from farms_amphibious.model.convention import AmphibiousConvention


class AmphibiousOptions(Options):
    """Simulation options"""

    def __init__(self, **kwargs):
        super(AmphibiousOptions, self).__init__()
        self.morphology = AmphibiousMorphologyOptions(**kwargs.pop('morphology'))
        self.spawn = AmphibiousSpawnOptions(**kwargs.pop('spawn'))
        self.physics = AmphibiousPhysicsOptions(**kwargs.pop('physics'))
        self.control = AmphibiousControlOptions(**kwargs.pop('control'))
        self.collect_gps = kwargs.pop('collect_gps')
        self.show_hydrodynamics = kwargs.pop('show_hydrodynamics')
        self.transition = kwargs.pop('transition')
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

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
        options['spawn'] = kwargs.pop(
            'spawn',
            AmphibiousSpawnOptions.from_options(kwargs)
        )
        options['physics'] = kwargs.pop(
            'physics',
            AmphibiousPhysicsOptions.from_options(kwargs)
        )
        if 'control' in kwargs:
            options['control'] = kwargs.pop('control')
        else:
            options['control'] = AmphibiousControlOptions.from_options(kwargs)
            options['control'].defaults_from_morphology(
                options['morphology'],
                kwargs
            )
        options['collect_gps'] = kwargs.pop('collect_gps', False)
        options['show_hydrodynamics'] = kwargs.pop('show_hydrodynamics', False)
        options['transition'] = kwargs.pop('transition', False)
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))
        return cls(**options)


class AmphibiousMorphologyOptions(Options):
    """Amphibious morphology options"""

    def __init__(self, **kwargs):
        super(AmphibiousMorphologyOptions, self).__init__()
        self.mesh_directory = kwargs.pop('mesh_directory')
        self.density = kwargs.pop('density')
        self.n_joints_body = kwargs.pop('n_joints_body')
        self.n_dof_legs = kwargs.pop('n_dof_legs')
        self.n_legs = kwargs.pop('n_legs')
        self.links = kwargs.pop('links')
        self.joints = kwargs.pop('joints')
        self.feet = kwargs.pop('feet')
        self.links_swimming = kwargs.pop('links_swimming')
        self.links_no_collisions = kwargs.pop('links_no_collisions')
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['mesh_directory'] = kwargs.pop('mesh_directory', '')
        options['density'] = kwargs.pop('density', 1000.0)
        options['n_joints_body'] = kwargs.pop('n_joints_body', 11)
        options['n_dof_legs'] = kwargs.pop('n_dof_legs', 4)
        options['n_legs'] = kwargs.pop('n_legs', 4)
        convention = AmphibiousConvention(**options)
        options['links'] = kwargs.pop('links', [
            convention.bodylink2name(i)
            for i in range(options['n_joints_body']+1)
        ] + [
            convention.leglink2name(leg_i, side_i, link_i)
            for leg_i in range(options['n_legs']//2)
            for side_i in range(2)
            for link_i in range(options['n_dof_legs'])
        ])
        options['joints'] = kwargs.pop('joints', [
            convention.bodyjoint2name(i)
            for i in range(options['n_joints_body'])
        ] + [
            convention.legjoint2name(leg_i, side_i, joint_i)
            for leg_i in range(options['n_legs']//2)
            for side_i in range(2)
            for joint_i in range(options['n_dof_legs'])
        ])
        options['feet'] = kwargs.pop('feet', [
            convention.leglink2name(
                leg_i=leg_i,
                side_i=side_i,
                joint_i=options['n_dof_legs']-1
            )
            for leg_i in range(options['n_legs']//2)
            for side_i in range(2)
        ])
        options['links_swimming'] = kwargs.pop('links_swimming', [
            convention.bodylink2name(body_i)
            for body_i in range(options['n_joints_body']+1)
        ])
        options['links_no_collisions'] = kwargs.pop('links_no_collisions', [
            convention.bodylink2name(body_i)
            for body_i in range(1, options['n_joints_body'])
        ] + [
            convention.leglink2name(leg_i, side_i, joint_i)
            for leg_i in range(options['n_legs']//2)
            for side_i in range(2)
            for joint_i in range(options['n_dof_legs']-1)
        ])
        return cls(**options)

    def n_joints(self):
        """Number of joints"""
        return self.n_joints_body + self.n_legs*self.n_dof_legs

    def n_joints_legs(self):
        """Number of legs joints"""
        return self.n_legs*self.n_dof_legs

    def n_links_body(self):
        """Number of body links"""
        return self.n_joints_body + 1

    def n_links(self):
        """Number of links"""
        return self.n_links_body() + self.n_joints_legs()


class AmphibiousSpawnOptions(Options):
    """Amphibious spawn options"""

    def __init__(self, **kwargs):
        super(AmphibiousSpawnOptions, self).__init__()
        self.position = kwargs.pop('position')
        self.orientation = kwargs.pop('orientation')
        self.velocity_lin = kwargs.pop('velocity_lin')
        self.velocity_ang = kwargs.pop('velocity_ang')
        self.joints_positions = kwargs.pop('joints_positions')
        self.joints_velocities = kwargs.pop('joints_velocities')
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        # Position in [m]
        options['position'] = kwargs.pop('spawn_position', [0, 0, 0.1])
        # Orientation in [rad] (Euler angles)
        options['orientation'] = kwargs.pop('spawn_orientation', [0, 0, 0])
        # Linear velocity in [m/s]
        options['velocity_lin'] = kwargs.pop('spawn_velocity_lin', [0, 0, 0])
        # Angular velocity in [rad/s] (Euler angles)
        options['velocity_ang'] = kwargs.pop('spawn_velocity_ang', [0, 0, 0])
        # Joints positions
        options['joints_positions'] = kwargs.pop('joints_positions', None)
        options['joints_velocities'] = kwargs.pop('joints_velocities', None)
        return cls(**options)


class AmphibiousPhysicsOptions(Options):
    """Amphibious physics options"""

    def __init__(self, **kwargs):
        super(AmphibiousPhysicsOptions, self).__init__()
        self.viscous = kwargs.pop('viscous')
        self.resistive = kwargs.pop('resistive')
        self.viscous_coefficients = kwargs.pop('viscous_coefficients')
        self.resistive_coefficients = kwargs.pop('resistive_coefficients')
        self.sph = kwargs.pop('sph')
        self.buoyancy = kwargs.pop('buoyancy')
        self.water_surface = kwargs.pop('water_surface')
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['viscous'] = kwargs.pop('viscous', False)
        options['resistive'] = kwargs.pop('resistive', False)
        options['viscous_coefficients'] = kwargs.pop(
            'viscous_coefficients',
            None
        )
        options['resistive_coefficients'] = kwargs.pop(
            'resistive_coefficients',
            None
        )
        options['sph'] = kwargs.pop('sph', False)
        options['buoyancy'] = kwargs.pop(
            'buoyancy',
            (options['resistive'] or options['viscous']) and not options['sph']
        )
        options['water_surface'] = kwargs.pop(
            'water_surface',
            options['viscous'] or options['resistive'] or options['sph']
        )
        return cls(**options)


class AmphibiousControlOptions(Options):
    """Amphibious control options"""

    def __init__(self, **kwargs):
        super(AmphibiousControlOptions, self).__init__()
        self.drives = AmphibiousDrives(**kwargs.pop('drives'))
        self.kinematics_file = kwargs.pop('kinematics_file')
        if not self.kinematics_file:
            self.network = AmphibiousNetworkOptions(**kwargs.pop('network'))
            self.joints = AmphibiousJointsOptions(**kwargs.pop('joints'))
            self.sensors = kwargs.pop('sensors')
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['kinematics_file'] = kwargs.pop('kinematics_file', '')
        options['drives'] = kwargs.pop(
            'drives',
            AmphibiousDrives.from_options(kwargs)
        )
        options['network'] = kwargs.pop(
            'network',
            AmphibiousNetworkOptions.from_options(kwargs)
        )
        options['joints'] = kwargs.pop(
            'joints',
            AmphibiousJointsOptions.from_options(kwargs)
        )
        options['sensors'] = kwargs.pop(
            'sensors',
            None
        )
        return cls(**options)

    def defaults_from_morphology(self, morphology, kwargs):
        """Defaults from morphology"""
        if self.joints.gain_amplitude is None:
            self.joints.gain_amplitude = (
                {joint: 1 for joint in morphology.joints}
            )
        if self.joints.gain_offset is None:
            self.joints.gain_offset = (
                {joint: 1 for joint in morphology.joints}
            )
        if self.joints.offsets is None:
            self.joints.offsets = (
                {joint: 0 for joint in morphology.joints}
            )
        if self.network.drive_init is None:
            self.network.drive_init = (
                [2, 0]
            )
        if self.network.state_init is None:
            self.network.state_init = (
                AmphibiousNetworkOptions.default_state_init(
                    morphology.n_joints(),
                ).tolist()
            )
        if self.network.osc_frequencies is None:
            self.network.osc_frequencies = (
                AmphibiousNetworkOptions.default_osc_frequencies(morphology)
            )
        if self.network.osc_amplitudes is None:
            self.network.osc_amplitudes = (
                AmphibiousNetworkOptions.default_osc_amplitudes(
                    morphology,
                    body_amplitude=kwargs.pop('body_stand_amplitude', 0.3),
                    legs_amplitudes=kwargs.pop(
                        'legs_amplitude',
                        [np.pi/4, np.pi/32, np.pi/4, np.pi/8]
                    ),
                )
            )
        if self.network.osc_rates is None:
            self.network.osc_rates = (
                AmphibiousNetworkOptions.default_osc_rates(morphology)
            )
        if self.network.osc2osc is None:
            self.network.osc2osc = (
                AmphibiousNetworkOptions.default_osc2osc(
                    morphology,
                    kwargs.pop('weight_osc_body', 1e3),
                    kwargs.pop(
                        'body_phase_bias',
                        2*np.pi/morphology.n_joints_body
                    ),
                    kwargs.pop('weight_osc_legs_internal', 1e3),
                    kwargs.pop('weight_osc_legs_opposite', 1e0),
                    kwargs.pop('weight_osc_legs_following', 1e0),
                    kwargs.pop('weight_osc_legs2body', 3e1),
                    kwargs.pop('leg_phase_follow', np.pi),
                )
            )
        if self.network.contact2osc is None:
            self.network.contact2osc = (
                AmphibiousNetworkOptions.default_contact2osc(
                    morphology,
                    kwargs.pop('weight_sens_contact_e', 2e0),
                    kwargs.pop('weight_sens_contact_i', -2e0),
                )
            )
        if self.network.hydro2osc is None:
            self.network.hydro2osc = (
                AmphibiousNetworkOptions.default_hydro2osc(
                    morphology,
                    kwargs.pop('weight_sens_hydro_freq', -1),
                    kwargs.pop('weight_sens_hydro_amp', 1),
                )
            )
        self.update(
            morphology.n_joints_body,
            morphology.n_dof_legs
        )

    def update(self, n_joints_body, n_dof_legs):
        """Update"""
        self.joints.update(n_joints_body, n_dof_legs)
        # self.network.oscillators.update(n_joints_body, n_dof_legs)


class AmphibiousDrives(Options):
    """Amphibious drives"""

    def __init__(self, **kwargs):
        super(AmphibiousDrives, self).__init__()
        self.forward = kwargs.pop('forward')
        self.turning = kwargs.pop('turning')
        self.left = kwargs.pop('left')
        self.right = kwargs.pop('right')
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['forward'] = kwargs.pop('drive_forward', 2)
        options['turning'] = kwargs.pop('drive_turn', 0)
        options['left'] = kwargs.pop('drive_left', 0)
        options['right'] = kwargs.pop('drive_right', 0)
        return cls(**options)


class AmphibiousNetworkOptions(Options):
    """Amphibious network options"""

    def __init__(self, **kwargs):
        super(AmphibiousNetworkOptions, self).__init__()
        # oscillators = kwargs.pop('oscillators')
        # self.oscillators = AmphibiousOscillatorOptions(**oscillators)

        # State
        self.state_init = kwargs.pop('state_init', None)

        # Nodes
        self.osc_nodes = kwargs.pop('osc_nodes', None)
        self.osc_frequencies = kwargs.pop('osc_frequencies', None)
        self.osc_rates = kwargs.pop('osc_rates', None)
        self.osc_amplitudes = kwargs.pop('osc_amplitudes', None)
        self.drive_nodes = kwargs.pop('drive_nodes', None)
        self.drive_init = kwargs.pop('drive_init', None)
        self.contacts_nodes = kwargs.pop('contacts_nodes', None)
        self.hydro_nodes = kwargs.pop('hydro_nodes', None)

        # Connections
        self.osc2osc = kwargs.pop('osc2osc', None)
        self.drive2osc = kwargs.pop('drive2osc', None)
        self.contact2osc = kwargs.pop('contact2osc', None)
        self.hydro2osc = kwargs.pop('hydro2osc', None)

        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        # options['oscillators'] = kwargs.pop(
        #     'oscillators',
        #     AmphibiousOscillatorOptions.from_options(kwargs)
        # )
        for option in [
                'osc_nodes',
                'osc_nodes',
                'osc_frequencies',
                'osc_rates',
                'osc_amplitudes',
                'drive_nodes',
                'drive_init',
                'contacts_nodes',
                'hydro_nodes',
                'osc2osc',
                'drive2osc',
                'contact2osc',
                'hydro2osc',
        ]:
            options[option] = kwargs.pop(option, None)
        return cls(**options)

    @staticmethod
    def default_state_init(n_joints):
        """Default state"""
        return 1e-3*np.arange(5*n_joints)

    @staticmethod
    def default_osc_frequencies(morphology):
        """Walking parameters"""
        n_oscillators = 2*(morphology.n_joints())
        convention = AmphibiousConvention(**morphology)
        n_oscillators = 2*(morphology.n_joints())
        frequencies = [None]*n_oscillators
        for joint_i in range(morphology.n_joints_body):
            for side in range(2):
                frequencies[convention.bodyosc2index(joint_i, side=side)] = {
                    'gain': 2*np.pi*0.2,
                    'bias': 2*np.pi*0.3,
                    'low': 1,
                    'high': 5,
                    'saturation': 0,
                }
        for joint_i in range(morphology.n_dof_legs):
            for leg_i in range(morphology.n_legs//2):
                for side_i in range(2):
                    for side in range(2):
                        frequencies[convention.legosc2index(
                            leg_i,
                            side_i,
                            joint_i,
                            side=side,
                        )] = {
                            'gain': 2*np.pi*0.2,
                            'bias': 2*np.pi*0.0,
                            'low': 1,
                            'high': 3,
                            'saturation': 0,
                        }
        return frequencies

    @staticmethod
    def default_osc_amplitudes(morphology, body_amplitude, legs_amplitudes):
        """Walking parameters"""
        convention = AmphibiousConvention(**morphology)
        n_oscillators = 2*(morphology.n_joints())
        amplitudes = [None]*n_oscillators
        # Body ampltidudes
        for joint_i in range(morphology.n_joints_body):
            for side in range(2):
                amplitudes[convention.bodyosc2index(joint_i, side=side)] = {
                    'gain': 0.25*body_amplitude,
                    'bias': 0.5*body_amplitude,
                    'low': 1,
                    'high': 5,
                    'saturation': 0,
                }
        # Legs ampltidudes
        for joint_i in range(morphology.n_dof_legs):
            amplitude = legs_amplitudes[joint_i]
            for leg_i in range(morphology.n_legs//2):
                for side_i in range(2):
                    for side in range(2):
                        amplitudes[convention.legosc2index(
                            leg_i,
                            side_i,
                            joint_i,
                            side=side,
                        )] = {
                            'gain': 0.5*amplitude,
                            'bias': 0*amplitude,
                            'low': 1,
                            'high': 3,
                            'saturation': 0,
                        }
        # pylog.debug('Amplitudes along body: abs({})'.format(amplitudes[:11]))
        return amplitudes

    @staticmethod
    def default_osc_rates(morphology):
        """Walking parameters"""
        n_oscillators = 2*(morphology.n_joints())
        rates = 10*np.ones(n_oscillators)
        return rates.tolist()

    @staticmethod
    def default_osc2osc(
            morphology,
            weight_body2body,
            phase_body2body,
            weight_intralimb,
            weight_interlimb_opposite,
            weight_interlimb_following,
            weight_limb2body,
            phase_limb_follow,
    ):
        """Default oscillartors to oscillators connectivity"""
        connectivity = []
        n_body_joints = morphology.n_joints_body
        # body_amplitude = connectivity_options.weight_osc_body
        # phase_diff = connectivity_options.body_phase_bias
        # legs_amplitude_internal = connectivity_options.weight_osc_legs_internal
        # legs_amplitude_opposite = connectivity_options.weight_osc_legs_opposite
        # legs_amplitude_following = connectivity_options.weight_osc_legs_following
        # legs2body_amplitude = connectivity_options.weight_osc_legs2body
        # phase_follow = connectivity_options.leg_phase_follow

        # Body
        convention = AmphibiousConvention(**morphology)
        for i in range(n_body_joints):
            for sides in [[1, 0], [0, 1]]:
                connectivity.append({
                    'in': convention.bodyosc2index(joint_i=i, side=sides[0]),
                    'out': convention.bodyosc2index(joint_i=i, side=sides[1]),
                    'weight': weight_body2body,
                    'phase_bias': np.pi,
                })
        for i in range(n_body_joints-1):
            for side in range(2):
                for osc, phase in [
                        [[i+1, i], phase_body2body],
                        [[i, i+1], -phase_body2body]
                ]:
                    connectivity.append({
                        'in': convention.bodyosc2index(joint_i=osc[0], side=side),
                        'out': convention.bodyosc2index(joint_i=osc[1], side=side),
                        'weight': weight_body2body,
                        'phase_bias': phase,
                    })

        # Legs (internal)
        for leg_i in range(morphology.n_legs//2):
            for side_i in range(2):
                _options = {
                    'leg_i': leg_i,
                    'side_i': side_i
                }
                # X - X
                for joint_i in range(morphology.n_dof_legs):
                    for sides in [[1, 0], [0, 1]]:
                        connectivity.append({
                            'in': convention.legosc2index(
                                **_options,
                                joint_i=joint_i,
                                side=sides[0]
                            ),
                            'out': convention.legosc2index(
                                **_options,
                                joint_i=joint_i,
                                side=sides[1]
                            ),
                            'weight': weight_intralimb,
                            'phase_bias': np.pi,
                        })

                # Following
                internal_connectivity = []
                if morphology.n_dof_legs > 1:
                    # 0 - 1
                    internal_connectivity.extend([
                        [[1, 0], 0, 0.5*np.pi],
                        [[0, 1], 0, -0.5*np.pi],
                        [[1, 0], 1, 0.5*np.pi],
                        [[0, 1], 1, -0.5*np.pi],
                    ])
                if morphology.n_dof_legs > 2:
                    # 0 - 2
                    internal_connectivity.extend([
                        [[2, 0], 0, 0],
                        [[0, 2], 0, 0],
                        [[2, 0], 1, 0],
                        [[0, 2], 1, 0],
                    ])
                if morphology.n_dof_legs > 3:
                    # 1 - 3
                    internal_connectivity.extend([
                        [[3, 1], 0, 0],
                        [[1, 3], 0, 0],
                        [[3, 1], 1, 0],
                        [[1, 3], 1, 0],
                    ])
                for joints, side, phase in internal_connectivity:
                    connectivity.append({
                        'in': convention.legosc2index(
                            **_options,
                            joint_i=joints[0],
                            side=side,
                        ),
                        'out': convention.legosc2index(
                            **_options,
                            joint_i=joints[1],
                            side=side,
                        ),
                        'weight': weight_intralimb,
                        'phase_bias': phase,
                    })

        # Opposite leg interaction
        for leg_i in range(morphology.n_legs//2):
            for joint_i in range(morphology.n_dof_legs):
                for side in range(2):
                    _options = {
                        'joint_i': joint_i,
                        'side': side
                    }
                    for sides in [[1, 0], [0, 1]]:
                        connectivity.append({
                            'in': convention.legosc2index(
                                leg_i=leg_i,
                                side_i=sides[0],
                                **_options
                            ),
                            'out': convention.legosc2index(
                                leg_i=leg_i,
                                side_i=sides[1],
                                **_options
                            ),
                            'weight': weight_interlimb_opposite,
                            'phase_bias': np.pi,
                        })

        # Following leg interaction
        for leg_pre in range(morphology.n_legs//2-1):
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
                            'in': convention.legosc2index(
                                leg_i=legs[0],
                                **_options
                            ),
                            'out': convention.legosc2index(
                                leg_i=legs[1],
                                **_options
                            ),
                            'weight': weight_interlimb_following,
                            'phase_bias': phase,
                        })

        # Body-legs interaction
        for leg_i in range(morphology.n_legs//2):
            for side_i in range(2):
                for i in range(n_body_joints):  # [0, 1, 7, 8, 9, 10]
                    for side_leg in range(2): # Muscle facing front/back
                        for lateral in range(2):
                            walk_phase = (
                                # i*2*np.pi/(n_body_joints-1)+0.5*np.pi
                                i*2*np.pi/(n_body_joints-1)+0.5*np.pi
                                # 0
                                # if np.cos(i*2*np.pi/(n_body_joints-1)) < 0
                                # else np.pi
                            )
                            # Forelimbs
                            connectivity.append({
                                'in': convention.bodyosc2index(
                                    joint_i=i,
                                    side=(side_i+lateral)%2
                                ),
                                'out': convention.legosc2index(
                                    leg_i=leg_i,
                                    side_i=side_i,
                                    joint_i=0,
                                    side=(side_i+side_leg)%2
                                ),
                                'weight': weight_limb2body,
                                'phase_bias': (
                                    walk_phase
                                    + np.pi*(side_i+1)
                                    + lateral*np.pi
                                    + side_leg*np.pi
                                    + leg_i*np.pi
                                ),
                            })
        return connectivity

    @staticmethod
    def default_contact2osc(morphology, weight_e, weight_i):
        """Default contact sensors to oscillators connectivity"""
        connectivity = []
        convention = AmphibiousConvention(**morphology)
        for leg_i in range(morphology.n_legs//2):
            for side_i in range(2):
                for joint_i in range(morphology.n_dof_legs):
                    for side_o in range(2):
                        for sensor_leg_i in range(morphology.n_legs//2):
                            for sensor_side_i in range(2):
                                weight = (
                                    weight_e
                                    if (
                                        (leg_i == sensor_leg_i)
                                        != (side_i == sensor_side_i)
                                    )
                                    else weight_i
                                )
                                connectivity.append({
                                    'in': convention.legosc2index(
                                        leg_i=leg_i,
                                        side_i=side_i,
                                        joint_i=joint_i,
                                        side=side_o
                                    ),
                                    'out': convention.contactleglink2index(
                                        leg_i=sensor_leg_i,
                                        side_i=sensor_side_i
                                    ),
                                    'weight': weight,
                                })
        return connectivity

    @staticmethod
    def default_hydro2osc(morphology, weight_frequency, weight_amplitude):
        """Default hydrodynamics sensors to oscillators connectivity"""
        connectivity = []
        # morphology.n_legs
        convention = AmphibiousConvention(**morphology)
        for joint_i in range(morphology.n_joints_body):
            for side_osc in range(2):
                connectivity.append({
                    'in': convention.bodyosc2index(
                        joint_i=joint_i,
                        side=side_osc
                    ),
                    'out': joint_i+1,
                    'weight_frequency': weight_frequency,
                    'weight_amplitude': weight_amplitude,
                })
        return connectivity


# class AmphibiousOscillatorOptions(Options):
#     """Amphibious oscillator options

#     Includes frequencies, amplitudes rates and nominal amplitudes

#     """

#     def __init__(self, **kwargs):
#         super(AmphibiousOscillatorOptions, self).__init__()
#         self.body_head_amplitude = kwargs.pop('body_head_amplitude')
#         self.body_tail_amplitude = kwargs.pop('body_tail_amplitude')
#         self.body_stand_amplitude = kwargs.pop('body_stand_amplitude')
#         self.legs_amplitudes = kwargs.pop('legs_amplitudes')
#         self.body_stand_shift = kwargs.pop('body_stand_shift')
#         self.body_nominal_amplitudes = kwargs.pop('body_nominal_amplitudes')
#         self.legs_nominal_amplitudes = kwargs.pop('legs_nominal_amplitudes')
#         self.body_freqs = kwargs.pop('body_freqs')
#         self.legs_freqs = kwargs.pop('legs_freqs')
#         if kwargs:
#             raise Exception('Unknown kwargs: {}'.format(kwargs))

#     @classmethod
#     def from_options(cls, kwargs):
#         """From options"""
#         options = {}
#         options['body_head_amplitude'] = kwargs.pop('body_head_amplitude', 0)
#         options['body_tail_amplitude'] = kwargs.pop('body_tail_amplitude', 0)
#         options['body_stand_amplitude'] = kwargs.pop('body_stand_amplitude', 0.2)
#         options['legs_amplitudes'] = kwargs.pop(
#             'legs_amplitude',
#             [np.pi/4, np.pi/32, np.pi/4, np.pi/8]
#         )
#         options['body_stand_shift'] = kwargs.pop('body_stand_shift', np.pi/4)
#         options['body_nominal_amplitudes'] = kwargs.pop(
#             'body_nominal_amplitudes',
#             None
#         )
#         options['legs_nominal_amplitudes'] = kwargs.pop(
#             'legs_nominal_amplitudes',
#             None
#         )

#         # Frequencies
#         options['body_freqs'] = [
#             [0, 0],
#             [1, 0],
#             [1, 1.5],
#             [5, 4],
#             [5, 0],
#             [6, 0]
#         ]
#         options['legs_freqs'] = [
#             [0, 0],
#             [1, 0],
#             [1, 0.5],
#             [3, 1.5],
#             [3, 0],
#             [6, 0]
#         ]
#         return cls(**options)

#     # def update(self, n_joints_body, n_dof_legs):
#     #     """Update all"""
#     #     self.set_body_nominal_amplitudes(n_joints_body)
#     #     self.set_legs_nominal_amplitudes(n_dof_legs)

#     def get_body_stand_amplitude(self):
#         """Body stand amplitude"""
#         return self.body_stand_amplitude

#     def set_body_stand_amplitude(self, value, n_joints_body):
#         """Body stand amplitude"""
#         self.body_stand_amplitude = value
#         self.set_body_nominal_amplitudes(n_joints_body)

#     def set_body_stand_shift(self, value, n_joints_body):
#         """Body stand shift"""
#         self.body_stand_shift = value
#         self.set_body_nominal_amplitudes(n_joints_body)

#     def set_body_nominal_amplitudes(self, n_joints_body):
#         """Set body nominal amplitudes"""
#         self.body_nominal_amplitudes = [
#             [
#                 [0, 0.3*self.body_stand_amplitude  # float(0.3*self.body_stand_amplitude*np.sin(
#                     # 2*np.pi*joint_i/n_joints_body - self.body_stand_shift))
#                 ],
#                 [3, self.body_stand_amplitude  # float(self.body_stand_amplitude*np.sin(
#                     # 2*np.pi*joint_i/n_joints_body - self.body_stand_shift))
#                 ],
#                 [3, 0.1*joint_i/n_joints_body],
#                 [5, 0.1*joint_i/n_joints_body+0.5],
#                 [5, 0],
#                 [6, 0],
#             ]
#             for joint_i in range(n_joints_body)
#         ]

#     def get_legs_amplitudes(self):
#         """Body legs amplitude"""
#         return self.legs_amplitudes

#     def set_legs_amplitudes(self, values, n_dof_legs):
#         """Body legs amplitude"""
#         self.legs_amplitudes = values
#         self.set_legs_nominal_amplitudes(n_dof_legs)

#     def set_legs_nominal_amplitudes(self, n_dof_legs):
#         """Set legs nominal amplitudes"""
#         self.legs_nominal_amplitudes = [
#             [
#                 [0, 0],
#                 [1, 0],
#                 [1, 0.7*self.legs_amplitudes[joint_i]],
#                 [3, self.legs_amplitudes[joint_i]],
#                 [3, 0],
#                 [6, 0],
#             ]
#             for joint_i in range(n_dof_legs)
#         ]


class AmphibiousJointsOptions(Options):
    """Amphibious joints options"""

    def __init__(self, **kwargs):
        super(AmphibiousJointsOptions, self).__init__()
        self._legs_offsets = kwargs.pop('_legs_offsets')
        self._legs_offsets_swimming = kwargs.pop('_legs_offsets_swimming')
        self.gain_amplitude = kwargs.pop('gain_amplitude')
        self.gain_offset = kwargs.pop('gain_offset')
        # Joints offsets
        self.offsets = kwargs.pop('offsets')
        self.legs_offsets = kwargs.pop('legs_offsets')
        self._body_offset = kwargs.pop('_body_offset')
        self.body_offsets = kwargs.pop('body_offsets')
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['_legs_offsets'] = kwargs.pop(
            'legs_offsets_walking',
            [0, np.pi/32, 0, np.pi/8]
        )
        options['_legs_offsets_swimming'] = kwargs.pop(
            'legs_offsets_swimming',
            [-2*np.pi/5, 0, 0, 0]
        )
        options['gain_amplitude'] = kwargs.pop('gain_amplitude', None)
        options['gain_offset'] = kwargs.pop('gain_offset', None)
        options['offsets'] = kwargs.pop('joints_offsets', None)
        # Joints offsets
        options['legs_offsets'] = None
        options['_body_offset'] = 0
        options['body_offsets'] = None
        return cls(**options)

    def update(self, n_joints_body, n_dof_legs):
        """Update"""
        self.update_body_offsets(n_joints_body)
        self.update_legs_offsets(n_dof_legs)

    def get_legs_offsets(self):
        """Get legs offsets"""
        return self._legs_offsets

    def set_legs_offsets(self, values, n_dof_legs):
        """Set legs offsets"""
        self._legs_offsets = values
        self.update_legs_offsets(n_dof_legs)

    def update_legs_offsets(self, n_dof_legs):
        """Set legs joints offsets"""
        self.legs_offsets = [
            [
                [0, self._legs_offsets_swimming[joint_i]],
                [1, self._legs_offsets_swimming[joint_i]],
                [1, self._legs_offsets[joint_i]],
                [3, self._legs_offsets[joint_i]],
                [3, self._legs_offsets_swimming[joint_i]],
                [6, self._legs_offsets_swimming[joint_i]]
            ]
            for joint_i in range(n_dof_legs)
        ]

    def set_body_offsets(self, value, n_joints_body):
        """Set body offsets"""
        self._body_offset = value
        self.update_body_offsets(n_joints_body)

    def update_body_offsets(self, n_joints_body):
        """Set body joints offsets"""
        self.body_offsets = [
            [
                [0, self._body_offset],
                [6, self._body_offset]
            ]
            for joint_i in range(n_joints_body)
        ]
