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
            raise Exception("Unknown kwargs: {}".format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['morphology'] = kwargs.pop(
            "morphology",
            AmphibiousMorphologyOptions.from_options(kwargs)
        )
        options['spawn'] = kwargs.pop(
            "spawn",
            AmphibiousSpawnOptions.from_options(kwargs)
        )
        options['physics'] = kwargs.pop(
            "physics",
            AmphibiousPhysicsOptions.from_options(kwargs)
        )
        if 'control' in kwargs:
            options['control'] = kwargs.pop('control')
        else:
            if 'body_phase_bias' not in kwargs:
                kwargs['body_phase_bias'] = (
                    2*np.pi/options['morphology'].n_joints_body
                )
            options['control'] = AmphibiousControlOptions.from_options(kwargs)
            options['control'].network.update(
                options['morphology'].n_joints_body,
                options['morphology'].n_dof_legs
            )
            options['control'].network.joints.gain_amplitude = (
                [1 for _ in range(options['morphology'].n_joints())]
            )
            options['control'].network.joints.gain_offset = (
                [1 for _ in range(options['morphology'].n_joints())]
            )

        options['collect_gps'] = kwargs.pop("collect_gps", False)
        options['show_hydrodynamics'] = kwargs.pop("show_hydrodynamics", False)
        options['transition'] = kwargs.pop("transition", False)
        if kwargs:
            raise Exception("Unknown kwargs: {}".format(kwargs))
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
            raise Exception("Unknown kwargs: {}".format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['mesh_directory'] = kwargs.pop('mesh_directory', '')
        options['density'] = kwargs.pop("density", 1000.0)
        options['n_joints_body'] = kwargs.pop("n_joints_body", 11)
        options['n_dof_legs'] = kwargs.pop("n_dof_legs", 4)
        options['n_legs'] = kwargs.pop("n_legs", 4)
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
            raise Exception("Unknown kwargs: {}".format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        # Position in [m]
        options['position'] = kwargs.pop("spawn_position", [0, 0, 0.1])
        # Orientation in [rad] (Euler angles)
        options['orientation'] = kwargs.pop("spawn_orientation", [0, 0, 0])
        # Linear velocity in [m/s]
        options['velocity_lin'] = kwargs.pop("spawn_velocity_lin", [0, 0, 0])
        # Angular velocity in [rad/s] (Euler angles)
        options['velocity_ang'] = kwargs.pop("spawn_velocity_ang", [0, 0, 0])
        # Joints positions
        options['joints_positions'] = kwargs.pop("joints_positions", None)
        options['joints_velocities'] = kwargs.pop("joints_velocities", None)
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
            raise Exception("Unknown kwargs: {}".format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['viscous'] = kwargs.pop("viscous", False)
        options['resistive'] = kwargs.pop("resistive", False)
        options['viscous_coefficients'] = kwargs.pop(
            "viscous_coefficients",
            None
        )
        options['resistive_coefficients'] = kwargs.pop(
            "resistive_coefficients",
            None
        )
        options['sph'] = kwargs.pop("sph", False)
        options['buoyancy'] = kwargs.pop(
            "buoyancy",
            (options['resistive'] or options['viscous']) and not options['sph']
        )
        options['water_surface'] = kwargs.pop(
            "water_surface",
            options['viscous'] or options['resistive'] or options['sph']
        )
        return cls(**options)


class AmphibiousControlOptions(Options):
    """Amphibious control options"""

    def __init__(self, **kwargs):
        super(AmphibiousControlOptions, self).__init__()
        self.kinematics_file = kwargs.pop('kinematics_file')
        self.drives = AmphibiousDrives(**kwargs.pop('drives'))
        self.network = AmphibiousNetworkOptions(**kwargs.pop('network'))
        if kwargs:
            raise Exception("Unknown kwargs: {}".format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['kinematics_file'] = kwargs.pop(
            "kinematics_file",
            ""
        )
        options['drives'] = kwargs.pop(
            "drives",
            AmphibiousDrives.from_options(kwargs)
        )
        # self.joints_controllers = kwargs.pop(
        #     "joints_controllers",
        #     AmphibiousJointsControllers(**options)
        # )
        options['network'] = kwargs.pop(
            "network",
            AmphibiousNetworkOptions.from_options(kwargs)
        )
        return cls(**options)


class AmphibiousDrives(Options):
    """Amphibious drives"""

    def __init__(self, **kwargs):
        super(AmphibiousDrives, self).__init__()
        self.forward = kwargs.pop('forward')
        self.turning = kwargs.pop('turning')
        self.left = kwargs.pop('left')
        self.right = kwargs.pop('right')
        if kwargs:
            raise Exception("Unknown kwargs: {}".format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['forward'] = kwargs.pop('drive_forward', 2)
        options['turning'] = kwargs.pop('drive_turn', 0)
        options['left'] = kwargs.pop('drive_left', 0)
        options['right'] = kwargs.pop('drive_right', 0)
        return cls(**options)


# class AmphibiousJointsControllers(Options):
#     """Amphibious joints controllers"""

#     def __init__(self, **kwargs):
#         super(AmphibiousJointsControllers, self).__init__()
#         self.body_p = kwargs.pop("body_p", 1e-1)
#         self.body_d = kwargs.pop("body_d", 1e0)
#         self.body_f = kwargs.pop("body_f", 1e1)
#         self.legs_p = kwargs.pop("legs_p", 1e-1)
#         self.legs_d = kwargs.pop("legs_d", 1e0)
#         self.legs_f = kwargs.pop("legs_f", 1e1)


class AmphibiousNetworkOptions(Options):
    """Amphibious network options"""

    def __init__(self, **kwargs):
        super(AmphibiousNetworkOptions, self).__init__()
        oscillators = kwargs.pop('oscillators')
        connectivity = kwargs.pop('connectivity')
        joints = kwargs.pop('joints')
        self.oscillators = AmphibiousOscillatorOptions(**oscillators)
        self.connectivity = AmphibiousConnectivityOptions(**connectivity)
        self.joints = AmphibiousJointsOptions(**joints)
        self.sensors = kwargs.pop('sensors')
        if kwargs:
            raise Exception("Unknown kwargs: {}".format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['oscillators'] = kwargs.pop(
            "oscillators",
            AmphibiousOscillatorOptions.from_options(kwargs)
        )
        options['connectivity'] = kwargs.pop(
            "connectivity",
            AmphibiousConnectivityOptions.from_options(kwargs)
        )
        options['joints'] = kwargs.pop(
            "joints",
            AmphibiousJointsOptions.from_options(kwargs)
        )
        options['sensors'] = kwargs.pop(
            "sensors",
            None
        )
        return cls(**options)

    def update(self, n_joints_body, n_dof_legs):
        """Update"""
        self.oscillators.update(n_joints_body, n_dof_legs)
        self.joints.update(n_joints_body, n_dof_legs)


class AmphibiousOscillatorOptions(Options):
    """Amphibious oscillator options

    Includes frequencies, amplitudes rates and nominal amplitudes

    """

    def __init__(self, **kwargs):
        super(AmphibiousOscillatorOptions, self).__init__()
        self.body_head_amplitude = kwargs.pop('body_head_amplitude')
        self.body_tail_amplitude = kwargs.pop('body_tail_amplitude')
        self._body_stand_amplitude = kwargs.pop('_body_stand_amplitude')
        self._legs_amplitudes = kwargs.pop('_legs_amplitudes')
        self._body_stand_shift = kwargs.pop('_body_stand_shift')
        self.body_nominal_amplitudes = kwargs.pop('body_nominal_amplitudes')
        self.legs_nominal_amplitudes = kwargs.pop('legs_nominal_amplitudes')
        self.body_freqs = kwargs.pop('body_freqs')
        self.legs_freqs = kwargs.pop('legs_freqs')
        if kwargs:
            raise Exception("Unknown kwargs: {}".format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['body_head_amplitude'] = kwargs.pop("body_head_amplitude", 0)
        options['body_tail_amplitude'] = kwargs.pop("body_tail_amplitude", 0)
        options['_body_stand_amplitude'] = kwargs.pop("body_stand_amplitude", 0.2)
        options['_legs_amplitudes'] = kwargs.pop(
            "legs_amplitude",
            [0.8, np.pi/32, np.pi/4, np.pi/8]
        )
        options['_body_stand_shift'] = kwargs.pop("body_stand_shift", np.pi/4)
        options['body_nominal_amplitudes'] = kwargs.pop(
            'body_nominal_amplitudes',
            None
        )
        options['legs_nominal_amplitudes'] = kwargs.pop(
            'legs_nominal_amplitudes',
            None
        )

        # Frequencies
        options['body_freqs'] = [
            [0, 0],
            [1, 0],
            [1, 1.5],
            [5, 4],
            [5, 0],
            [6, 0]
        ]
        options['legs_freqs'] = [
            [0, 0],
            [1, 0],
            [1, 0.5],
            [3, 1.5],
            [3, 0],
            [6, 0]
        ]
        return cls(**options)

    def update(self, n_joints_body, n_dof_legs):
        """Update all"""
        self.set_body_nominal_amplitudes(n_joints_body)
        self.set_legs_nominal_amplitudes(n_dof_legs)

    def get_body_stand_amplitude(self):
        """Body stand amplitude"""
        return self._body_stand_amplitude

    def set_body_stand_amplitude(self, value, n_joints_body):
        """Body stand amplitude"""
        self._body_stand_amplitude = value
        self.set_body_nominal_amplitudes(n_joints_body)

    def set_body_stand_shift(self, value, n_joints_body):
        """Body stand shift"""
        self._body_stand_shift = value
        self.set_body_nominal_amplitudes(n_joints_body)

    def set_body_nominal_amplitudes(self, n_joints_body):
        """Set body nominal amplitudes"""
        body_stand_shift = np.pi/4
        self.body_nominal_amplitudes = [
            [
                [0, float(0.3*self._body_stand_amplitude*np.sin(
                    2*np.pi*joint_i/n_joints_body - body_stand_shift
                ))],
                [3, float(self._body_stand_amplitude*np.sin(
                    2*np.pi*joint_i/n_joints_body - body_stand_shift
                ))],
                [3, 0.1*joint_i/n_joints_body],
                [5, 0.6*joint_i/n_joints_body+0.2],
                [5, 0],
                [6, 0]
            ]
            for joint_i in range(n_joints_body)
        ]

    def get_legs_amplitudes(self):
        """Body legs amplitude"""
        return self._legs_amplitudes

    def set_legs_amplitudes(self, values, n_dof_legs):
        """Body legs amplitude"""
        self._legs_amplitudes = values
        self.set_legs_nominal_amplitudes(n_dof_legs)

    def set_legs_nominal_amplitudes(self, n_dof_legs):
        """Set legs nominal amplitudes"""
        self.legs_nominal_amplitudes = [
            [
                [0, 0],
                [1, 0],
                [1, 0.7*self._legs_amplitudes[joint_i]],
                [3, self._legs_amplitudes[joint_i]],
                [3, 0],
                [6, 0]
            ]
            for joint_i in range(n_dof_legs)
        ]


class AmphibiousConnectivityOptions(Options):
    """Amphibious connectivity options"""

    def __init__(self, **kwargs):
        super(AmphibiousConnectivityOptions, self).__init__()
        self.body_head_amplitude = kwargs.pop("body_head_amplitude")
        self.body_phase_bias = kwargs.pop("body_phase_bias")
        self.leg_phase_follow = kwargs.pop("leg_phase_follow")
        self.weight_osc_body = kwargs.pop('weight_osc_body')
        self.weight_osc_legs_internal = kwargs.pop('weight_osc_legs_internal')
        self.weight_osc_legs_opposite = kwargs.pop('weight_osc_legs_opposite')
        self.weight_osc_legs_following = kwargs.pop('weight_osc_legs_following')
        self.weight_osc_legs2body = kwargs.pop('weight_osc_legs2body')
        self.weight_sens_contact_i = kwargs.pop('weight_sens_contact_i')
        self.weight_sens_contact_e = kwargs.pop('weight_sens_contact_e')
        self.weight_sens_hydro_freq = kwargs.pop('weight_sens_hydro_freq')
        self.weight_sens_hydro_amp = kwargs.pop('weight_sens_hydro_amp')
        if kwargs:
            raise Exception("Unknown kwargs: {}".format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['body_head_amplitude'] = kwargs.pop("body_head_amplitude", 0)
        options['body_phase_bias'] = kwargs.pop(
            "body_phase_bias",
            None
        )
        options['leg_phase_follow'] = kwargs.pop(
            "leg_phase_follow",
            np.pi
        )
        options['weight_osc_body'] = 1e3
        options['weight_osc_legs_internal'] = 1e3
        options['weight_osc_legs_opposite'] = 1e0
        options['weight_osc_legs_following'] = 1e0
        options['weight_osc_legs2body'] = kwargs.pop('w_legs2body', 3e1)
        options['weight_sens_contact_i'] = kwargs.pop('w_sens_contact_i', -2e0)
        options['weight_sens_contact_e'] = kwargs.pop('w_sens_contact_e', 2e0)
        options['weight_sens_hydro_freq'] = kwargs.pop('w_sens_hyfro_freq', -1)
        options['weight_sens_hydro_amp'] = kwargs.pop('w_sens_hydro_amp', 1)
        return cls(**options)


class AmphibiousJointsOptions(Options):
    """Amphibious joints options"""

    def __init__(self, **kwargs):
        super(AmphibiousJointsOptions, self).__init__()
        self._legs_offsets = kwargs.pop("_legs_offsets")
        self._legs_offsets_swimming = kwargs.pop("_legs_offsets_swimming")
        self.gain_amplitude = kwargs.pop("gain_amplitude")
        self.gain_offset = kwargs.pop("gain_offset")
        # Joints offsets
        self.legs_offsets = kwargs.pop("legs_offsets")
        self._body_offset = kwargs.pop("_body_offset")
        self.body_offsets = kwargs.pop("body_offsets")
        if kwargs:
            raise Exception("Unknown kwargs: {}".format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['_legs_offsets'] = kwargs.pop(
            "legs_offsets_walking",
            [0, np.pi/32, 0, np.pi/8]
        )
        options['_legs_offsets_swimming'] = kwargs.pop(
            "legs_offsets_swimming",
            [-2*np.pi/5, 0, 0, 0]
        )
        options['gain_amplitude'] = kwargs.pop("gain_amplitude", None)
        options['gain_offset'] = kwargs.pop("gain_offset", None)
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
