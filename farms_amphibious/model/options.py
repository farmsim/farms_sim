"""Animat options"""

from enum import Enum
import numpy as np

from farms_data.options import Options
from farms_amphibious.model.convention import AmphibiousConvention


class SpawnLoader(Enum):
    """Spawn loader"""
    FARMS = 0
    PYBULLET = 1


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
        self.mass_multiplier = kwargs.pop('mass_multiplier')
        self.n_joints_body = kwargs.pop('n_joints_body')
        self.n_dof_legs = kwargs.pop('n_dof_legs')
        self.n_legs = kwargs.pop('n_legs')
        self.links = kwargs.pop('links')
        self.feet = kwargs.pop('feet')
        self.links_swimming = kwargs.pop('links_swimming')
        self.links_no_collisions = kwargs.pop('links_no_collisions')
        self.links_friction_lateral = kwargs.pop('links_friction_lateral')
        self.links_friction_spinning = kwargs.pop('links_friction_spinning')
        self.links_friction_rolling = kwargs.pop('links_friction_rolling')
        self.joints = kwargs.pop('joints')
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['mesh_directory'] = kwargs.pop('mesh_directory', '')
        options['density'] = kwargs.pop('density', 1000.0)
        options['mass_multiplier'] = kwargs.pop('mass_multiplier', 1)
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
        options['links_no_collisions'] = kwargs.pop('links_no_collisions', (
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
        options['links_friction_lateral'] = kwargs.pop(
            'links_friction_lateral',
            [1 for link in options['links']]
        )
        options['links_friction_spinning'] = kwargs.pop(
            'links_friction_spinning',
            [0 for link in options['links']]
        )
        options['links_friction_rolling'] = kwargs.pop(
            'links_friction_rolling',
            [0 for link in options['links']]
        )
        options['joints'] = kwargs.pop('joints', [
            convention.bodyjoint2name(i)
            for i in range(options['n_joints_body'])
        ] + [
            convention.legjoint2name(leg_i, side_i, joint_i)
            for leg_i in range(options['n_legs']//2)
            for side_i in range(2)
            for joint_i in range(options['n_dof_legs'])
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
        self.loader = kwargs.pop('loader')
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
        # Loader
        options['loader'] = kwargs.pop('spawn_loader', SpawnLoader.PYBULLET)
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
        self.drag = kwargs.pop('drag')
        self.drag_coefficients = kwargs.pop('drag_coefficients')
        self.sph = kwargs.pop('sph')
        self.buoyancy = kwargs.pop('buoyancy')
        self.water_surface = kwargs.pop('water_surface')
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['drag'] = kwargs.pop('drag', False)
        options['drag_coefficients'] = kwargs.pop(
            'drag_coefficients',
            None
        )
        options['sph'] = kwargs.pop('sph', False)
        options['buoyancy'] = kwargs.pop(
            'buoyancy',
            options['drag'] and not options['sph']
        )
        options['water_surface'] = kwargs.pop(
            'water_surface',
            options['drag'] or options['sph']
        )
        return cls(**options)


class AmphibiousControlOptions(Options):
    """Amphibious control options"""

    def __init__(self, **kwargs):
        super(AmphibiousControlOptions, self).__init__()
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
        if self.network.drives_init is None:
            self.network.drives_init = (
                [2, 0]
            )
        if self.network.state_init is None:
            self.network.state_init = (
                AmphibiousNetworkOptions.default_state_init(
                    morphology,
                ).tolist()
            )
        if self.network.oscillators is None:
            self.network.oscillators = (
                AmphibiousNetworkOptions.default_oscillators(
                    morphology.n_joints(),
                )
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
                        'legs_amplitudes',
                        [np.pi/4, np.pi/32, np.pi/4, np.pi/8]
                    ),
                )
            )
        if self.network.osc_rates is None:
            self.network.osc_rates = (
                AmphibiousNetworkOptions.default_osc_rates(morphology)
            )
        if self.network.osc_modular_phases is None:
            self.network.osc_modular_phases = (
                AmphibiousNetworkOptions.default_osc_modular_phases(
                    morphology=morphology,
                    phases=kwargs.pop('modular_phases', np.zeros(4)),
                )
            )
        if self.network.osc_modular_amplitudes is None:
            self.network.osc_modular_amplitudes = (
                AmphibiousNetworkOptions.default_osc_modular_amplitudes(
                    morphology=morphology,
                    amplitudes=kwargs.pop('modular_amplitudes', np.zeros(4)),
                )
            )
        if self.network.osc2osc is None:
            self.network.osc2osc = (
                AmphibiousNetworkOptions.default_osc2osc(
                    morphology,
                    kwargs.pop('weight_osc_body', 1e0),
                    kwargs.pop(
                        'body_phase_bias',
                        2*np.pi/morphology.n_joints_body
                    ),
                    kwargs.pop('weight_osc_legs_internal', 3e1),
                    kwargs.pop('weight_osc_legs_opposite', 1e1),
                    kwargs.pop('weight_osc_legs_following', 1e1),
                    kwargs.pop('weight_osc_legs2body', 3e1),
                    kwargs.pop('leg_phase_follow', np.pi),
                    kwargs.pop('body_stand_shift', 0.5*np.pi),
                )
            )
        if self.network.joint2osc is None:
            self.network.joint2osc = []
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
        self.joints.defaults_from_morphology(morphology, kwargs)


class AmphibiousNetworkOptions(Options):
    """Amphibious network options"""

    def __init__(self, **kwargs):
        super(AmphibiousNetworkOptions, self).__init__()

        # State
        self.state_init = kwargs.pop('state_init', None)
        self.drives_init = kwargs.pop('drives_init', None)

        # Oscillators
        self.oscillators = kwargs.pop('oscillators', None)
        self.osc_frequencies = kwargs.pop('osc_frequencies', None)
        self.osc_amplitudes = kwargs.pop('osc_amplitudes', None)
        self.osc_rates = kwargs.pop('osc_rates', None)
        self.osc_modular_phases = kwargs.pop('osc_modular_phases', None)
        self.osc_modular_amplitudes = kwargs.pop('osc_modular_amplitudes', None)
        self.joints_output = kwargs.pop('joints_output', None)

        # Connections
        self.osc2osc = kwargs.pop('osc2osc', None)
        self.drive2osc = kwargs.pop('drive2osc', None)
        self.joint2osc = kwargs.pop('joint2osc', None)
        self.contact2osc = kwargs.pop('contact2osc', None)
        self.hydro2osc = kwargs.pop('hydro2osc', None)

        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        for option in [
                # State
                'state_init',
                'drives_init',
                # Oscillators
                'oscillators',
                'osc_frequencies',
                'osc_amplitudes',
                'osc_rates',
                'osc_modular_phases',
                'osc_modular_amplitudes',
                # Connections
                'osc2osc',
                'drive2osc',
                'joint2osc',
                'contact2osc',
                'hydro2osc',
                # Joints output
                'joints_output',
        ]:
            options[option] = kwargs.pop(option, None)
        return cls(**options)

    @staticmethod
    def default_state_init(morphology):
        """Default state"""
        convention = AmphibiousConvention(**morphology)
        state = np.zeros(5*morphology.n_joints())
        phases_init_body = np.linspace(2*np.pi, 0, morphology.n_joints_body)
        for joint_i in range(morphology.n_joints_body):
            for side_osc in range(2):
                state[convention.bodyosc2index(
                    joint_i,
                    side=side_osc,
                )] = (
                    phases_init_body[joint_i]
                    + (0 if side_osc else np.pi)
                )
        phases_init_legs = [3*np.pi/2, 0, 3*np.pi/2, 0]
        for joint_i in range(morphology.n_dof_legs):
            for leg_i in range(morphology.n_legs//2):
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
        state += 1e-3*np.arange(5*morphology.n_joints())
        return state

    @staticmethod
    def default_oscillators(n_joints):
        """Default oscillator names"""
        return ['O_{}'.format(i) for i in range(n_joints)]

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
                            'gain': 0,
                            'bias': amplitude,
                            'low': 1,
                            'high': 3,
                            'saturation': 0,
                        }
        return amplitudes

    @staticmethod
    def default_osc_rates(morphology):
        """Walking parameters"""
        n_oscillators = 2*(morphology.n_joints())
        rates = 10*np.ones(n_oscillators)
        return rates.tolist()

    @staticmethod
    def default_osc_modular_phases(morphology, phases):
        """Default"""
        convention = AmphibiousConvention(**morphology)
        n_oscillators = 2*(morphology.n_joints())
        values = 0*np.ones(n_oscillators)
        for joint_i in range(morphology.n_dof_legs):
            phase = phases[joint_i]
            for leg_i in range(morphology.n_legs//2):
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
    def default_osc_modular_amplitudes(morphology, amplitudes):
        """Default"""
        convention = AmphibiousConvention(**morphology)
        n_oscillators = 2*(morphology.n_joints())
        values = 0*np.ones(n_oscillators)
        for joint_i in range(morphology.n_dof_legs):
            amplitude = amplitudes[joint_i]
            for leg_i in range(morphology.n_legs//2):
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
            morphology,
            weight_body2body,
            phase_body2body,
            weight_intralimb,
            weight_interlimb_opposite,
            weight_interlimb_following,
            weight_limb2body,
            phase_limb_follow,
            body_stand_shift,
    ):
        """Default oscillators to oscillators connectivity"""
        connectivity = []
        n_body_joints = morphology.n_joints_body

        # Body
        convention = AmphibiousConvention(**morphology)
        if weight_body2body != 0:
            for i in range(n_body_joints):
                for sides in [[1, 0], [0, 1]]:
                    connectivity.append({
                        'in': convention.bodyosc2index(
                            joint_i=i,
                            side=sides[0]
                        ),
                        'out': convention.bodyosc2index(
                            joint_i=i,
                            side=sides[1]
                        ),
                        'type': 'OSC2OSC',
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
                            'in': convention.bodyosc2index(
                                joint_i=osc[0],
                                side=side
                            ),
                            'out': convention.bodyosc2index(
                                joint_i=osc[1],
                                side=side
                            ),
                            'type': 'OSC2OSC',
                            'weight': weight_body2body,
                            'phase_bias': phase,
                        })

        # Legs (internal)
        if weight_intralimb != 0:
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
                                'type': 'OSC2OSC',
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
                            'type': 'OSC2OSC',
                            'weight': weight_intralimb,
                            'phase_bias': phase,
                        })

        # Opposite leg interaction
        if weight_interlimb_opposite != 0:
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
                                'type': 'OSC2OSC',
                                'weight': weight_interlimb_opposite,
                                'phase_bias': np.pi,
                            })

        # Following leg interaction
        if weight_interlimb_following != 0:
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
                                'type': 'OSC2OSC',
                                'weight': weight_interlimb_following,
                                'phase_bias': phase,
                            })

        # Body-legs interaction
        if weight_limb2body != 0:
            for leg_i in range(morphology.n_legs//2):
                for side_i in range(2):
                    for i in range(n_body_joints):  # [0, 1, 7, 8, 9, 10]
                        for side_leg in range(2): # Muscle facing front/back
                            for lateral in range(1):
                                walk_phase = (
                                    # i*2*np.pi/(n_body_joints-1)+0.5*np.pi
                                    i*2*np.pi/(n_body_joints-1) + body_stand_shift
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
                                    'type': 'OSC2OSC',
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
                                if weight != 0:
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
                                        'type': 'REACTION2FREQ',
                                        'weight': weight,
                                    })
        return connectivity

    @staticmethod
    def default_hydro2osc(morphology, weight_frequency, weight_amplitude):
        """Default hydrodynamics sensors to oscillators connectivity"""
        connectivity = []
        convention = AmphibiousConvention(**morphology)
        for joint_i in range(morphology.n_joints_body):
            for side_osc in range(2):
                if weight_frequency:
                    connectivity.append({
                        'in': convention.bodyosc2index(
                            joint_i=joint_i,
                            side=side_osc
                        ),
                        'out': joint_i+1,
                        'type': 'LATERAL2FREQ',
                        'weight': weight_frequency,
                    })
                if weight_amplitude:
                    connectivity.append({
                        'in': convention.bodyosc2index(
                            joint_i=joint_i,
                            side=side_osc
                        ),
                        'out': joint_i+1,
                        'type': 'LATERAL2AMP',
                        'weight': weight_amplitude,
                    })
        return connectivity


class AmphibiousJointsOptions(Options):
    """Amphibious joints options"""

    def __init__(self, **kwargs):
        super(AmphibiousJointsOptions, self).__init__()
        self.offsets = kwargs.pop('offsets')
        self.rates = kwargs.pop('rates')
        self.gain_amplitude = kwargs.pop('gain_amplitude')
        self.gain_offset = kwargs.pop('gain_offset')
        self.offsets_bias = kwargs.pop('offsets_bias')
        self.max_torques = kwargs.pop('max_torques')
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['offsets'] = kwargs.pop('offsets', None)
        options['rates'] = kwargs.pop('rates', None)
        options['gain_amplitude'] = kwargs.pop('gain_amplitude', None)
        options['gain_offset'] = kwargs.pop('gain_offset', None)
        options['offsets_bias'] = kwargs.pop('offsets_bias', None)
        options['max_torques'] = kwargs.pop('max_torques', None)
        return cls(**options)

    def defaults_from_morphology(self, morphology, kwargs):
        """Joints """
        convention = AmphibiousConvention(**morphology)
        if self.offsets is None:
            self.offsets = [None]*morphology.n_joints()
            # Turning body
            for joint_i in range(morphology.n_joints_body):
                for side_i in range(2):
                    self.offsets[convention.bodyjoint2index(joint_i=joint_i)] = {
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
                [-2*np.pi/5, 0, 0, 0]
            )
            leg_turn_gain = kwargs.pop(
                'leg_turn_gain',
                [-1, 1]
            )
            leg_side_turn_gain = kwargs.pop(
                'leg_side_turn_gain',
                [-1, 1]
            )
            leg_joint_turn_gain = kwargs.pop(
                'leg_joint_turn_gain',
                [1, 0, 0, 0]
            )
            for leg_i in range(morphology.n_legs//2):
                for side_i in range(2):
                    for joint_i in range(morphology.n_dof_legs):
                        self.offsets[convention.legjoint2index(
                            leg_i=leg_i,
                            side_i=side_i,
                            joint_i=joint_i,
                        )] = {
                            'gain': (
                                leg_turn_gain[leg_i]
                                *leg_side_turn_gain[side_i]
                                *leg_joint_turn_gain[joint_i]
                            ),
                            'bias': legs_offsets_walking[joint_i],
                            'low': 1,
                            'high': 3,
                            'saturation': legs_offsets_swimming[joint_i],
                        }
        if self.rates is None:
            self.rates = [5]*morphology.n_joints()
        if self.gain_amplitude is None:
            self.gain_amplitude = (
                {joint: 1 for joint in morphology.joints}
            )
        if self.gain_offset is None:
            self.gain_offset = (
                {joint: 1 for joint in morphology.joints}
            )
        if self.offsets_bias is None:
            self.offsets_bias = (
                {joint: 0 for joint in morphology.joints}
            )
        if self.max_torques is None:
            self.max_torques = (
                {joint: 100 for joint in morphology.joints}
            )
