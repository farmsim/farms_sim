"""Animat data"""

import numpy as np
from scipy import interpolate

from ..data.animat_data import (
    OscillatorNetworkState,
    AnimatData,
    NetworkParameters,
    DriveArray,
    Oscillators,
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


DTYPE = np.float64
ITYPE = np.uintc


class AmphibiousData(AnimatData):
    """Amphibious network parameter"""

    @classmethod
    def from_options(
            cls,
            initial_drive,
            initial_state,
            morphology,
            control,
            n_iterations
    ):
        """Default amphibious newtwork parameters"""
        state = OscillatorNetworkState.from_initial_state(
            initial_state,
            n_iterations,
        )
        network = NetworkParameters(
            drives=DriveArray.from_initial_drive(
                initial_drive,
                n_iterations,
            ),
            oscillators=Oscillators.from_options(
                control.network,
            ),
            osc_connectivity=OscillatorConnectivity.from_connectivity(
                control.network.osc2osc
            ),
            contacts_connectivity=ContactConnectivity.from_connectivity(
                control.network.contact2osc
            ),
            hydro_connectivity=HydroConnectivity.from_connectivity(
                control.network.hydro2osc,
            ),
        )
        joints = AmphibiousJointsArray.from_options(
            morphology,
            control,
        )
        sensors = SensorsData(
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


# class AmphibiousOscillatorArray(OscillatorArray):
#     """Oscillator array"""

#     @staticmethod
#     def set_options(morphology, oscillators, drives):
#         """Walking parameters"""
#         n_body = morphology.n_joints_body
#         n_dof_legs = morphology.n_dof_legs
#         n_legs = morphology.n_legs
#         convention = AmphibiousConvention(**morphology)
#         # n_oscillators = 2*(morphology.n_joints_body)
#         n_oscillators = 2*(morphology.n_joints())
#         data = np.array(oscillators.body_freqs, dtype=DTYPE)
#         freqs_body = 2*np.pi*np.ones(2*morphology.n_joints_body, dtype=DTYPE)*(
#             # oscillators.body_freqs.value(drives)
#             interpolate.interp1d(data[:, 0], data[:, 1])(drives.forward)
#         )
#         data = np.array(oscillators.legs_freqs, dtype=DTYPE)
#         freqs_legs = 2*np.pi*np.ones(2*morphology.n_joints_legs(), dtype=DTYPE)*(
#             # oscillators.legs_freqs.value(drives)
#             interpolate.interp1d(data[:, 0], data[:, 1])(drives.forward)
#         )
#         freqs = np.concatenate([freqs_body, freqs_legs])
#         rates = 10*np.ones(n_oscillators, dtype=DTYPE)
#         # Amplitudes
#         amplitudes = np.zeros(n_oscillators, dtype=DTYPE)
#         for i in range(n_body):
#             data = np.array(oscillators.body_nominal_amplitudes[i], dtype=DTYPE)
#             amplitudes[convention.bodyosc2index(i, side=0)] = (
#                 interpolate.interp1d(data[:, 0], data[:, 1])(drives.forward)
#             )
#             amplitudes[convention.bodyosc2index(i, side=1)] = (
#                 interpolate.interp1d(data[:, 0], data[:, 1])(drives.forward)
#             )
#         for i in range(n_dof_legs):
#             data = np.array(oscillators.legs_nominal_amplitudes[i], dtype=DTYPE)
#             interp = interpolate.interp1d(data[:, 0], data[:, 1])
#             for leg_i in range(n_legs//2):
#                 for side_i in range(2):
#                     for side in range(2):
#                         amplitudes[convention.legosc2index(
#                             leg_i,
#                             side_i,
#                             i,
#                             side=side
#                         )] = interp(drives.forward)
#         # pylog.debug('Amplitudes along body: abs({})'.format(amplitudes[:11]))
#         return np.abs(freqs), np.abs(rates), np.abs(amplitudes)

#     @classmethod
#     def from_options(cls, morphology, oscillators, drives):
#         """Default"""
#         freqs, rates, amplitudes = cls.set_options(
#             morphology,
#             oscillators,
#             drives
#         )
#         return cls.from_parameters(freqs, rates, amplitudes)

#     def update(self, morphology, oscillators, drives):
#         """Update from options

#         :param options: Animat options

#         """
#         freqs, _, amplitudes = self.set_options(
#             morphology,
#             oscillators,
#             drives,
#         )
#         self.freqs()[:] = freqs
#         self.amplitudes_desired()[:] = amplitudes


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
