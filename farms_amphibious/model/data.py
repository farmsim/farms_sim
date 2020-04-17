"""Animat data"""

import numpy as np
from scipy import interpolate

from ..data.animat_data import (
    OscillatorNetworkState,
    AnimatData,
    NetworkParameters,
    OscillatorArray,
    OscillatorConnectivity,
    ContactConnectivity,
    HydroConnectivity,
    JointsArray,
    SensorsData,
    ContactsArray,
    ProprioceptionArray,
    GpsArray,
    HydrodynamicsArray
)
from .convention import AmphibiousConvention


DTYPE = np.float64
ITYPE = np.uintc


class AmphibiousData(AnimatData):
    """Amphibious network parameter"""

    @classmethod
    def from_options(cls, state, morphology, control, n_iterations):
        """Default amphibious newtwork parameters"""
        network = NetworkParameters(
            oscillators=AmphibiousOscillatorArray.from_options(
                morphology,
                control.network.oscillators,
                control.drives,
            ),
            # osc_connectivity=AmphibiousOscillatorConnectivity.from_options(
            #     morphology,
            #     control.network.connectivity,
            # ),
            osc_connectivity=OscillatorConnectivity.from_connectivity(
                control.network.osc2osc
            ),
            # contacts_connectivity=AmphibiousContactsConnectivity.from_options(
            #     morphology,
            #     control.network.connectivity,
            # )
            contacts_connectivity=ContactConnectivity.from_connectivity(
                control.network.contact2osc
            ),
            # hydro_connectivity=AmphibiousHydroConnectivity.from_options(
            #     morphology,
            #     control.network.connectivity,
            # ),
            hydro_connectivity=HydroConnectivity.from_connectivity(
                control.network.hydro2osc,
            ),
        )
        joints = AmphibiousJointsArray.from_options(
            morphology,
            control,
        )
        sensors = SensorsData(
            # contacts=AmphibiousContactsArray.from_options(
            #     morphology.n_legs,
            #     n_iterations,
            # ),
            # proprioception=AmphibiousProprioceptionArray.from_options(
            #     morphology.n_joints(),
            #     n_iterations,
            # ),
            # gps=AmphibiousGpsArray.from_options(
            #     morphology.n_links(),
            #     n_iterations,
            # ),
            # hydrodynamics=AmphibiousHydrodynamicsArray.from_options(
            #     morphology.n_links_body(),
            #     n_iterations,
            # )
            contacts=ContactsArray.from_size(
                morphology.n_legs,
                n_iterations,
            ),
            proprioception=ProprioceptionArray.from_size(
                morphology.n_joints(),
                n_iterations,
            ),
            gps=GpsArray.from_size(
                morphology.n_links(),
                n_iterations,
            ),
            hydrodynamics=HydrodynamicsArray.from_size(
                morphology.n_links_body(),
                n_iterations,
            )
        )
        return cls(state, network, joints, sensors)


class AmphibiousOscillatorNetworkState(OscillatorNetworkState):
    """Network state"""

    @staticmethod
    def default_initial_state(morphology):
        """Default state"""
        n_joints = morphology.n_joints()
        return 1e-3*np.arange(5*n_joints, dtype=DTYPE) + np.concatenate([
            # 0*np.linspace(2*np.pi, 0, n_joints),
            np.zeros(n_joints, dtype=DTYPE),
            np.zeros(n_joints, dtype=DTYPE),
            np.zeros(2*n_joints, dtype=DTYPE),
            np.zeros(n_joints, dtype=DTYPE)
        ])

    @staticmethod
    def default_state(n_iterations, morphology):
        """Default state"""
        n_joints = morphology.n_joints()
        n_oscillators = 2*n_joints
        return AmphibiousOscillatorNetworkState.from_initial_state(
            initial_state=(
                AmphibiousOscillatorNetworkState.default_initial_state(
                    morphology
                )
            ),
            n_iterations=n_iterations,
            n_oscillators=n_oscillators
        )

    @classmethod
    def from_initial_state(cls, initial_state, n_iterations, n_oscillators):
        """From initial state"""
        state = np.zeros(
            [n_iterations, 2, np.shape(initial_state)[0]],
            dtype=DTYPE
        )
        state[0, 0, :] = np.array(initial_state, dtype=DTYPE)
        return cls(state, n_oscillators)


class AmphibiousOscillatorArray(OscillatorArray):
    """Oscillator array"""

    @staticmethod
    def set_options(morphology, oscillators, drives):
        """Walking parameters"""
        n_body = morphology.n_joints_body
        n_dof_legs = morphology.n_dof_legs
        n_legs = morphology.n_legs
        convention = AmphibiousConvention(**morphology)
        # n_oscillators = 2*(morphology.n_joints_body)
        n_oscillators = 2*(morphology.n_joints())
        data = np.array(oscillators.body_freqs, dtype=DTYPE)
        freqs_body = 2*np.pi*np.ones(2*morphology.n_joints_body, dtype=DTYPE)*(
            # oscillators.body_freqs.value(drives)
            interpolate.interp1d(data[:, 0], data[:, 1])(drives.forward)
        )
        data = np.array(oscillators.legs_freqs, dtype=DTYPE)
        freqs_legs = 2*np.pi*np.ones(2*morphology.n_joints_legs(), dtype=DTYPE)*(
            # oscillators.legs_freqs.value(drives)
            interpolate.interp1d(data[:, 0], data[:, 1])(drives.forward)
        )
        freqs = np.concatenate([freqs_body, freqs_legs])
        rates = 10*np.ones(n_oscillators, dtype=DTYPE)
        # Amplitudes
        amplitudes = np.zeros(n_oscillators, dtype=DTYPE)
        for i in range(n_body):
            data = np.array(oscillators.body_nominal_amplitudes[i], dtype=DTYPE)
            amplitudes[convention.bodyosc2index(i, side=0)] = (
                interpolate.interp1d(data[:, 0], data[:, 1])(drives.forward)
            )
            amplitudes[convention.bodyosc2index(i, side=1)] = (
                interpolate.interp1d(data[:, 0], data[:, 1])(drives.forward)
            )
        for i in range(n_dof_legs):
            data = np.array(oscillators.legs_nominal_amplitudes[i], dtype=DTYPE)
            interp = interpolate.interp1d(data[:, 0], data[:, 1])
            for leg_i in range(n_legs//2):
                for side_i in range(2):
                    for side in range(2):
                        amplitudes[convention.legosc2index(
                            leg_i,
                            side_i,
                            i,
                            side=side
                        )] = interp(drives.forward)
        # pylog.debug('Amplitudes along body: abs({})'.format(amplitudes[:11]))
        return np.abs(freqs), np.abs(rates), np.abs(amplitudes)

    @classmethod
    def from_options(cls, morphology, oscillators, drives):
        """Default"""
        freqs, rates, amplitudes = cls.set_options(
            morphology,
            oscillators,
            drives
        )
        return cls.from_parameters(freqs, rates, amplitudes)

    def update(self, morphology, oscillators, drives):
        """Update from options

        :param options: Animat options

        """
        freqs, _, amplitudes = self.set_options(
            morphology,
            oscillators,
            drives,
        )
        self.freqs()[:] = freqs
        self.amplitudes_desired()[:] = amplitudes


# class AmphibiousOscillatorConnectivity(OscillatorConnectivity):
#     """Connectivity array"""

#     @staticmethod
#     def set_options(morphology, connectivity_options, verbose=False):
#         """Walking parameters"""
#         # osc_options = control.network.oscillators
#         n_body_joints = morphology.n_joints_body
#         connectivity, weights, desired_phases = [], [], []
#         body_amplitude = connectivity_options.weight_osc_body
#         legs_amplitude_internal = connectivity_options.weight_osc_legs_internal
#         legs_amplitude_opposite = connectivity_options.weight_osc_legs_opposite
#         legs_amplitude_following = connectivity_options.weight_osc_legs_following
#         legs2body_amplitude = connectivity_options.weight_osc_legs2body

#         # # Amplitudes
#         # amplitudes = [
#         #     osc_options.body_stand_amplitude*np.sin(
#         #         2*np.pi*i/n_body_joints
#         #         - osc_options.body_stand_shift
#         #     )
#         #     for i in range(n_body_joints)
#         # ]

#         # Body
#         convention = AmphibiousConvention(**morphology)
#         for i in range(n_body_joints):
#             for sides in [[1, 0], [0, 1]]:
#                 connectivity.append([
#                     convention.bodyosc2index(joint_i=i, side=sides[0]),
#                     convention.bodyosc2index(joint_i=i, side=sides[1]),
#                 ])
#                 weights.append(body_amplitude)
#                 desired_phases.append(np.pi)
#         for i in range(n_body_joints-1):
#             # i - i+1
#             phase_diff = connectivity_options.body_phase_bias
#             phase_follow = connectivity_options.leg_phase_follow
#             # phase_diff = np.pi/11
#             for side in range(2):
#                 for osc, phase in [
#                         [[i+1, i], phase_diff],
#                         [[i, i+1], -phase_diff]
#                 ]:
#                     connectivity.append([
#                         convention.bodyosc2index(joint_i=osc[0], side=side),
#                         convention.bodyosc2index(joint_i=osc[1], side=side),
#                     ])
#                     weights.append(body_amplitude)
#                     desired_phases.append(phase)

#         # Legs (internal)
#         for leg_i in range(morphology.n_legs//2):
#             for side_i in range(2):
#                 _options = {
#                     'leg_i': leg_i,
#                     'side_i': side_i
#                 }
#                 # X - X
#                 for joint_i in range(morphology.n_dof_legs):
#                     for sides in [[1, 0], [0, 1]]:
#                         connectivity.append([
#                             convention.legosc2index(
#                                 **_options,
#                                 joint_i=joint_i,
#                                 side=sides[0]
#                             ),
#                             convention.legosc2index(
#                                 **_options,
#                                 joint_i=joint_i,
#                                 side=sides[1]
#                             ),
#                         ])
#                         weights.append(legs_amplitude_internal)
#                         desired_phases.append(np.pi)

#                 # Following
#                 internal_connectivity = []
#                 if morphology.n_dof_legs > 1:
#                     # 0 - 1
#                     internal_connectivity.extend([
#                         [[1, 0], 0, 0.5*np.pi],
#                         [[0, 1], 0, -0.5*np.pi],
#                         [[1, 0], 1, 0.5*np.pi],
#                         [[0, 1], 1, -0.5*np.pi],
#                     ])
#                 if morphology.n_dof_legs > 2:
#                     # 0 - 2
#                     internal_connectivity.extend([
#                         [[2, 0], 0, 0],
#                         [[0, 2], 0, 0],
#                         [[2, 0], 1, 0],
#                         [[0, 2], 1, 0],
#                     ])
#                 if morphology.n_dof_legs > 3:
#                     # 1 - 3
#                     internal_connectivity.extend([
#                         [[3, 1], 0, 0],
#                         [[1, 3], 0, 0],
#                         [[3, 1], 1, 0],
#                         [[1, 3], 1, 0],
#                     ])
#                 for joints, side, phase in internal_connectivity:
#                     connectivity.append([
#                         convention.legosc2index(
#                             **_options,
#                             joint_i=joints[0],
#                             side=side,
#                         ),
#                         convention.legosc2index(
#                             **_options,
#                             joint_i=joints[1],
#                             side=side,
#                         ),
#                     ])
#                     weights.append(legs_amplitude_internal)
#                     desired_phases.append(phase)

#         # Opposite leg interaction
#         for leg_i in range(morphology.n_legs//2):
#             for joint_i in range(morphology.n_dof_legs):
#                 for side in range(2):
#                     _options = {
#                         'joint_i': joint_i,
#                         'side': side
#                     }
#                     for sides in [[1, 0], [0, 1]]:
#                         connectivity.append([
#                             convention.legosc2index(
#                                 leg_i=leg_i,
#                                 side_i=sides[0],
#                                 **_options
#                             ),
#                             convention.legosc2index(
#                                 leg_i=leg_i,
#                                 side_i=sides[1],
#                                 **_options
#                             ),
#                         ])
#                         weights.append(legs_amplitude_opposite)
#                         desired_phases.append(np.pi)

#         # Following leg interaction
#         for leg_pre in range(morphology.n_legs//2-1):
#             for side_i in range(2):
#                 for side in range(2):
#                     _options = {
#                         'side_i': side_i,
#                         'side': side,
#                         'joint_i': 0,
#                     }
#                     for legs, phase in [
#                             [[leg_pre, leg_pre+1], phase_follow],
#                             [[leg_pre+1, leg_pre], -phase_follow],
#                     ]:
#                         connectivity.append([
#                             convention.legosc2index(
#                                 leg_i=legs[0],
#                                 **_options
#                             ),
#                             convention.legosc2index(
#                                 leg_i=legs[1],
#                                 **_options
#                             ),
#                         ])
#                         weights.append(legs_amplitude_following)
#                         desired_phases.append(phase)

#         # Body-legs interaction
#         for leg_i in range(morphology.n_legs//2):
#             for side_i in range(2):
#                 for i in range(n_body_joints):  # [0, 1, 7, 8, 9, 10]
#                     for side_leg in range(2): # Muscle facing front/back
#                         for lateral in range(2):
#                             walk_phase = (
#                                 0
#                                 if i in [0, 1, 7, 8, 9, 10]
#                                 else np.pi
#                             )
#                             # Forelimbs
#                             connectivity.append([
#                                 convention.bodyosc2index(
#                                     joint_i=i,
#                                     side=(side_i+lateral)%2
#                                 ),
#                                 convention.legosc2index(
#                                     leg_i=leg_i,
#                                     side_i=side_i,
#                                     joint_i=0,
#                                     side=(side_i+side_leg)%2
#                                 ),
#                             ])
#                             weights.append(legs2body_amplitude)
#                             desired_phases.append(
#                                 walk_phase
#                                 + np.pi*(side_i+1)
#                                 + lateral*np.pi
#                                 + side_leg*np.pi
#                                 + leg_i*np.pi
#                             )
#         if verbose:
#             with np.printoptions(
#                     suppress=True,
#                     precision=3,
#                     threshold=sys.maxsize
#             ):
#                 pylog.debug('Oscillator connectivity:\n{}'.format(
#                     np.array(connectivity, dtype=DTYPE)
#                 ))
#         return connectivity, weights, desired_phases

#     @classmethod
#     def from_options(cls, morphology, control):
#         """Parameters for walking"""
#         connectivity, weights, phases = cls.set_options(morphology, control)
#         return cls(
#             np.array(connectivity, dtype=ITYPE),
#             np.array(weights, dtype=DTYPE),
#             np.array(phases, dtype=DTYPE),
#         )

#     def update(self, morphology, control):
#         """Update from options

#         :param options: Animat options

#         """


class AmphibiousJointsArray(JointsArray):
    """Oscillator array"""

    @staticmethod
    def set_options(morphology, control):
        """Walking parameters"""
        j_options = control.joints
        n_body = morphology.n_joints_body
        n_dof_legs = morphology.n_dof_legs
        n_legs = morphology.n_legs
        n_joints = n_body + n_legs*n_dof_legs
        offsets = np.zeros(n_joints, dtype=DTYPE)
        # Body offset
        offsets[:n_body] = control.drives.turning
        # Legs walking/swimming
        for i in range(n_dof_legs):
            data = np.array(j_options.legs_offsets[i], dtype=DTYPE)
            interp = interpolate.interp1d(data[:, 0], data[:, 1])
            for leg_i in range(n_legs):
                offsets[n_body + leg_i*n_dof_legs + i] = (
                    interp(control.drives.forward)
                )
        # Turning legs
        for leg_i in range(n_legs//2):
            for side in range(2):
                offsets[n_body + 2*leg_i*n_dof_legs + side*n_dof_legs + 0] += (
                    control.drives.turning
                    *(1 if leg_i else -1)
                    *(1 if side else -1)
                )
        # Turning body
        for i in range(n_body):
            data = np.array(j_options.body_offsets[i], dtype=DTYPE)
            offsets[i] += (
                interpolate.interp1d(data[:, 0], data[:, 1])(
                    control.drives.forward
                )
            )
        rates = 5*np.ones(n_joints, dtype=DTYPE)
        return offsets, rates

    @classmethod
    def from_options(cls, morphology, control):
        """Parameters for walking"""
        offsets, rates = cls.set_options(morphology, control)
        return cls.from_parameters(offsets, rates)

    def update(self, morphology, control):
        """Update from options

        :param options: Animat options

        """
        offsets, _ = self.set_options(morphology, control)
        self.offsets()[:] = offsets


# class AmphibiousContactsArray(ContactsArray):
#     """Amphibious contacts sensors array"""

#     @classmethod
#     def from_options(cls, n_contacts, n_iterations):
#         """Default"""
#         contacts = np.zeros([n_iterations, n_contacts, 9], dtype=DTYPE)  # x, y, z
#         return cls(contacts)


# class AmphibiousContactsConnectivity(ContactConnectivity):
#     """Amphibious contacts connectivity array"""

#     @classmethod
#     def from_options(cls, morphology, connectivity_options, verbose=False):
#         """Default"""
#         connectivity, weights = [], []
#         # morphology.n_legs
#         convention = AmphibiousConvention(**morphology)
#         for leg_i in range(morphology.n_legs//2):
#             for side_i in range(2):
#                 for joint_i in range(morphology.n_dof_legs):
#                     for side_o in range(2):
#                         for sensor_leg_i in range(morphology.n_legs//2):
#                             for sensor_side_i in range(2):
#                                 weight = (
#                                     connectivity_options.weight_sens_contact_e
#                                     if (
#                                         (leg_i == sensor_leg_i)
#                                         != (side_i == sensor_side_i)
#                                     )
#                                     else connectivity_options.weight_sens_contact_i
#                                 )
#                                 connectivity.append([
#                                     convention.legosc2index(
#                                         leg_i=leg_i,
#                                         side_i=side_i,
#                                         joint_i=joint_i,
#                                         side=side_o
#                                     ),
#                                     convention.contactleglink2index(
#                                         leg_i=sensor_leg_i,
#                                         side_i=sensor_side_i
#                                     ),
#                                 ])
#                                 weights.append(weight)
#         if verbose:
#             pylog.debug('Contacts connectivity:\n{}'.format(
#                 np.array(connectivity, dtype=DTYPE)
#             ))
#         return cls(
#             np.array(connectivity, dtype=ITYPE),
#             np.array(weights, dtype=DTYPE),
#         )


# class AmphibiousHydroConnectivity(HydroConnectivity):
#     """Amphibious hydro connectivity array"""

#     @classmethod
#     def from_options(cls, morphology, connectivity_options, verbose=False):
#         """Default"""
#         connectivity = []
#         frequencies = []
#         amplitudes = []
#         # morphology.n_legs
#         convention = AmphibiousConvention(**morphology)
#         for joint_i in range(morphology.n_joints_body):
#             for side_osc in range(2):
#                 connectivity.append([
#                     convention.bodyosc2index(
#                         joint_i=joint_i,
#                         side=side_osc
#                     ),
#                     joint_i+1,
#                 ])
#                 frequencies.append(connectivity_options.weight_sens_hydro_freq)
#                 amplitudes.append(connectivity_options.weight_sens_hydro_amp)
#         if verbose:
#             pylog.debug('Hydro connectivity:\n{}'.format(
#                 np.array(connectivity, dtype=ITYPE)
#             ))
#         return cls(
#             connections=np.array(connectivity, dtype=ITYPE),
#             frequency=np.array(frequencies, dtype=DTYPE),
#             amplitude=np.array(amplitudes, dtype=DTYPE),
#         )


# class AmphibiousProprioceptionArray(ProprioceptionArray):
#     """Amphibious proprioception sensors array"""

#     @classmethod
#     def from_options(cls, n_joints, n_iterations):
#         """Default"""
#         proprioception = np.zeros([n_iterations, n_joints, 12], dtype=DTYPE)
#         return cls(proprioception)


# class AmphibiousGpsArray(GpsArray):
#     """Amphibious gps sensors array"""

#     @classmethod
#     def from_options(cls, n_links, n_iterations):
#         """Default"""
#         gps = np.zeros([n_iterations, n_links, 20], dtype=DTYPE)
#         return cls(gps)


# class AmphibiousHydrodynamicsArray(HydrodynamicsArray):
#     """Amphibious hydrodynamics sensors array"""

#     @classmethod
#     def from_options(cls, n_links, n_iterations):
#         """Default"""
#         hydrodynamics = np.zeros([n_iterations, n_links, 6], dtype=DTYPE)  # Fxyz, Mxyz
#         return cls(hydrodynamics)
