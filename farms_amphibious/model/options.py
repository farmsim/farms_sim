"""Animat options"""

import numpy as np
# from scipy import interpolate

from farms_bullet.simulation.options import Options
from farms_amphibious.model.convention import AmphibiousConvention


class AmphibiousOptions(Options):
    """Simulation options"""

    def __init__(self, **kwargs):
        super(AmphibiousOptions, self).__init__()
        self.morphology = kwargs.pop(
            "morphology",
            AmphibiousMorphologyOptions(kwargs)
        )
        self.spawn = kwargs.pop(
            "spawn",
            AmphibiousSpawnOptions(kwargs)
        )
        self.physics = kwargs.pop(
            "physics",
            AmphibiousPhysicsOptions(kwargs)
        )
        self.control = kwargs.pop(
            "control",
            AmphibiousControlOptions(self.morphology, kwargs)
        )
        self.collect_gps = kwargs.pop(
            "collect_gps",
            False
        )
        self.show_hydrodynamics = kwargs.pop(
            "show_hydrodynamics",
            False
        )
        self.transition = kwargs.pop(
            "transition",
            False
        )
        if kwargs:
            raise Exception("Unknown kwargs: {}".format(kwargs))


class AmphibiousMorphologyOptions(Options):
    """Amphibious morphology options"""

    def __init__(self, options):
        super(AmphibiousMorphologyOptions, self).__init__()
        self.mesh_directory = ""
        self.density = options.pop("density", 1000.0)
        self.n_joints_body = options.pop("n_joints_body", 11)
        self.n_dof_legs = options.pop("n_dof_legs", 4)
        self.n_legs = options.pop("n_legs", 4)
        convention = AmphibiousConvention(self)
        self.links = options.pop('links', [
            convention.bodylink2name(i)
            for i in range(self.n_links_body())
        ] + [
            convention.leglink2name(leg_i, side_i, link_i)
            for leg_i in range(self.n_legs//2)
            for side_i in range(2)
            for link_i in range(self.n_dof_legs)
        ])
        self.joints = options.pop('joints', [
            convention.bodyjoint2name(i)
            for i in range(self.n_joints_body)
        ] + [
            convention.legjoint2name(leg_i, side_i, joint_i)
            for leg_i in range(self.n_legs//2)
            for side_i in range(2)
            for joint_i in range(self.n_dof_legs)
        ])
        self.feet = options.pop('feet', [
            convention.leglink2name(
                leg_i=leg_i,
                side_i=side_i,
                joint_i=self.n_dof_legs-1
            )
            for leg_i in range(self.n_legs//2)
            for side_i in range(2)
        ])
        self.links_swimming = options.pop('links_swimming', [
            convention.bodylink2name(body_i)
            for body_i in range(self.n_links_body())
        ])
        self.links_no_collisions = options.pop('links_no_collisions', [
            convention.bodylink2name(body_i)
            for body_i in range(1, self.n_links_body()-1)
        ] + [
            convention.leglink2name(leg_i, side_i, joint_i)
            for leg_i in range(self.n_legs//2)
            for side_i in range(2)
            for joint_i in range(self.n_dof_legs-1)
        ])

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

    def __init__(self, options):
        super(AmphibiousSpawnOptions, self).__init__()
        # Position in [m]
        self.position = options.pop("spawn_position", [0, 0, 0.1])
        # Orientation in [rad] (Euler angles)
        self.orientation = options.pop("spawn_orientation", [0, 0, 0])
        # Linear velocity in [m/s]
        self.velocity_lin = options.pop("spawn_velocity_lin", [0, 0, 0])
        # Angular velocity in [rad/s] (Euler angles)
        self.velocity_ang = options.pop("spawn_velocity_ang", [0, 0, 0])
        # Joints positions
        self.joints_positions = options.pop("joints_positions", None)
        self.joints_velocities = options.pop("joints_velocities", None)


class AmphibiousPhysicsOptions(Options):
    """Amphibious physics options"""

    def __init__(self, options):
        super(AmphibiousPhysicsOptions, self).__init__()
        self.viscous = options.pop("viscous", False)
        self.resistive = options.pop("resistive", False)
        self.viscous_coefficients = options.pop(
            "viscous_coefficients",
            None
        )
        self.resistive_coefficients = options.pop(
            "resistive_coefficients",
            None
        )
        self.sph = options.pop("sph", False)
        self.buoyancy = options.pop(
            "buoyancy",
            (self.resistive or self.viscous) and not self.sph
        )
        self.water_surface = options.pop(
            "water_surface",
            self.viscous or self.resistive or self.sph
        )


class AmphibiousControlOptions(Options):
    """Amphibious control options"""

    def __init__(self, morphology, kwargs):
        super(AmphibiousControlOptions, self).__init__()
        self.kinematics_file = kwargs.pop(
            "kinematics_file",
            ""
        )
        self.drives = kwargs.pop(
            "drives",
            AmphibiousDrives(**kwargs)
        )
        # self.joints_controllers = kwargs.pop(
        #     "joints_controllers",
        #     AmphibiousJointsControllers(**kwargs)
        # )
        self.network = kwargs.pop(
            "network",
            AmphibiousNetworkOptions(morphology, kwargs)
        )


class AmphibiousDrives(Options):
    """Amphibious drives"""

    def __init__(self, **kwargs):
        super(AmphibiousDrives, self).__init__()
        self.forward = kwargs.pop("drive_forward", 2)
        self.turning = kwargs.pop("drive_turn", 0)
        self.left = kwargs.pop("drive_left", 0)
        self.right = kwargs.pop("drive_right", 0)


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

    def __init__(self, morphology, kwargs):
        super(AmphibiousNetworkOptions, self).__init__()
        self.oscillators = kwargs.pop(
            "oscillators",
            AmphibiousOscillatorOptions(morphology, kwargs)
        )
        self.connectivity = kwargs.pop(
            "connectivity",
            AmphibiousConnectivityOptions(morphology, kwargs)
        )
        self.joints = kwargs.pop(
            "joints",
            AmphibiousJointsOptions(morphology, kwargs)
        )
        self.sensors = kwargs.pop(
            "sensors",
            None
        )

    def update(self):
        """Update"""
        self.oscillators.update()
        self.joints.update()


class AmphibiousOscillatorOptions(Options):
    """Amphibious oscillator options

    Includes frequencies, amplitudes rates and nominal amplitudes

    """

    def __init__(self, morphology, kwargs):
        super(AmphibiousOscillatorOptions, self).__init__()
        self.n_joints_body = morphology.n_joints_body
        self.n_dof_legs = morphology.n_dof_legs
        self.body_head_amplitude = kwargs.pop("body_head_amplitude", 0)
        self.body_tail_amplitude = kwargs.pop("body_tail_amplitude", 0)
        self._body_stand_amplitude = kwargs.pop("body_stand_amplitude", 0.2)
        self._legs_amplitudes = kwargs.pop(
            "legs_amplitude",
            [0.8, np.pi/32, np.pi/4, np.pi/8]
        )
        self._body_stand_shift = kwargs.pop("body_stand_shift", np.pi/4)
        self.body_nominal_amplitudes = None
        self.legs_nominal_amplitudes = None
        self.update()

        # Frequencies
        self.body_freqs = [
            [0, 0],
            [1, 0],
            [1, 1.5],
            [5, 4],
            [5, 0],
            [6, 0]
        ]
        self.legs_freqs = [
            [0, 0],
            [1, 0],
            [1, 0.5],
            [3, 1.5],
            [3, 0],
            [6, 0]
        ]

    def update(self):
        """Update all"""
        self.set_body_nominal_amplitudes()
        self.set_legs_nominal_amplitudes()

    def get_body_stand_amplitude(self):
        """Body stand amplitude"""
        return self._body_stand_amplitude

    def set_body_stand_amplitude(self, value):
        """Body stand amplitude"""
        self._body_stand_amplitude = value
        self.set_body_nominal_amplitudes()

    def set_body_stand_shift(self, value):
        """Body stand shift"""
        self._body_stand_shift = value
        self.set_body_nominal_amplitudes()

    def set_body_nominal_amplitudes(self):
        """Set body nominal amplitudes"""
        n_body = self.n_joints_body
        body_stand_shift = np.pi/4
        self.body_nominal_amplitudes = [
            [
                [0, float(0.3*self._body_stand_amplitude*np.sin(
                    2*np.pi*joint_i/n_body - body_stand_shift
                ))],
                [3, float(self._body_stand_amplitude*np.sin(
                    2*np.pi*joint_i/n_body - body_stand_shift
                ))],
                [3, 0.1*joint_i/n_body],
                [5, 0.6*joint_i/n_body+0.2],
                [5, 0],
                [6, 0]
            ]
            for joint_i in range(self.n_joints_body)
        ]

    def get_legs_amplitudes(self):
        """Body legs amplitude"""
        return self._legs_amplitudes

    def set_legs_amplitudes(self, values):
        """Body legs amplitude"""
        self._legs_amplitudes = values
        self.set_legs_nominal_amplitudes()

    def set_legs_nominal_amplitudes(self):
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
            for joint_i in range(self.n_dof_legs)
        ]


class AmphibiousConnectivityOptions(Options):
    """Amphibious connectivity options"""

    def __init__(self, morphology, kwargs):
        super(AmphibiousConnectivityOptions, self).__init__()
        self.body_phase_bias = kwargs.pop(
            "body_phase_bias",
            2*np.pi/morphology.n_joints_body
        )
        self.leg_phase_follow = kwargs.pop(
            "leg_phase_follow",
            np.pi
        )
        self.weight_osc_body = 1e3
        self.weight_osc_legs_internal = 1e3
        self.weight_osc_legs_opposite = 1e0
        self.weight_osc_legs_following = 1e0
        self.weight_osc_legs2body = kwargs.pop('w_legs2body', 3e1)
        self.weight_sens_contact_i = kwargs.pop('w_sens_contact_i', -2e0)
        self.weight_sens_contact_e = kwargs.pop('w_sens_contact_e', 2e0)  # +3e-1
        self.weight_sens_hydro_freq = kwargs.pop('w_sens_hyfro_freq', -1)
        self.weight_sens_hydro_amp = kwargs.pop('w_sens_hydro_amp', 1)


class AmphibiousJointsOptions(Options):
    """Amphibious joints options"""

    def __init__(self, morphology, kwargs):
        super(AmphibiousJointsOptions, self).__init__()
        self.n_joints_body = morphology.n_joints_body
        self.n_dof_legs = morphology.n_dof_legs
        self._legs_offsets = kwargs.pop(
            "legs_offsets_walking",
            [0, np.pi/32, 0, np.pi/8]
        )
        self._legs_offsets_swimming = kwargs.pop(
            "legs_offsets_swimming",
            [-2*np.pi/5, 0, 0, 0]
        )
        self.gain_amplitude = kwargs.pop(
            "gain_amplitude",
            [1 for _ in range(morphology.n_joints())]
        )
        self.gain_offset = kwargs.pop(
            "gain_offset",
            [1 for _ in range(morphology.n_joints())]
        )
        # Joints offsets
        self.legs_offsets = None
        self.update_legs_offsets()
        self._body_offset = 0
        self.body_offsets = None
        self.update_body_offsets()

    def update(self):
        """Update"""
        self.update_body_offsets()
        self.update_legs_offsets()

    def get_legs_offsets(self):
        """Get legs offsets"""
        return self._legs_offsets

    def set_legs_offsets(self, values):
        """Set legs offsets"""
        self._legs_offsets = values
        self.update_legs_offsets()

    def update_legs_offsets(self):
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
            for joint_i in range(self.n_dof_legs)
        ]

    def set_body_offsets(self, value):
        """Set body offsets"""
        self._body_offset = value
        self.update_body_offsets()

    def update_body_offsets(self):
        """Set body joints offsets"""
        self.body_offsets = [
            [
                [0, self._body_offset],
                [6, self._body_offset]
            ]
            for joint_i in range(self.n_joints_body)
        ]
