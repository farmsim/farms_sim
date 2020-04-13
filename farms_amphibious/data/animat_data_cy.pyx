"""Animat data"""

import numpy as np
cimport numpy as np


cdef class AnimatDataCy:
    """Network parameter"""

    def __init__(self, state=None, network=None, joints=None, sensors=None):
        super(AnimatDataCy, self).__init__()
        self.state = state
        self.network = network
        self.joints = joints
        self.sensors = sensors


cdef class NetworkParametersCy:
    """Network parameter"""

    def __init__(
            self,
            oscillators,
            connectivity,
            contacts_connectivity,
            hydro_connectivity
    ):
        super(NetworkParametersCy, self).__init__()
        self.oscillators = oscillators
        self.connectivity = connectivity
        self.contacts_connectivity = contacts_connectivity
        self.hydro_connectivity = hydro_connectivity


cdef class OscillatorNetworkStateCy(NetworkArray3D):
    """Network state"""

    def __init__(self, state, n_oscillators, iteration=0):
        super(OscillatorNetworkStateCy, self).__init__(state)
        self.n_oscillators = n_oscillators
        self._iteration = iteration

    @classmethod
    def from_options(cls, state, animat_options, iteration=0):
        """From options"""
        return cls(
            state=state,
            n_oscillators=2*animat_options.morphology.n_joints(),
            iteration=iteration
        )

    @classmethod
    def from_solver(cls, solver, n_oscillators):
        """From solver"""
        return cls(solver.state, n_oscillators, solver.iteration)

    def phases(self, unsigned int iteration):
        """Phases"""
        return self.array[iteration, 0, :self.n_oscillators]

    def phases_all(self):
        """Phases"""
        return self.array[:, 0, :self.n_oscillators]

    def amplitudes(self, unsigned int iteration):
        """Amplitudes"""
        return self.array[iteration, 0, self.n_oscillators:]

    def amplitudes_all(self):
        """Phases"""
        return self.array[:, 0, self.n_oscillators:]

    def dphases(self, unsigned int iteration):
        """Phases derivative"""
        return self.array[iteration, 1, :self.n_oscillators]

    def damplitudes(self, unsigned int iteration):
        """Amplitudes derivative"""
        return self.array[iteration, 1, self.n_oscillators:]


cdef class OscillatorArrayCy(NetworkArray2D):
    """Oscillator array"""

    @classmethod
    def from_parameters(cls, freqs, rates, amplitudes):
        """From each parameter"""
        return cls(np.array([freqs, rates, amplitudes]))

    cpdef CTYPEv1 freqs(self):
        """Frequencies"""
        return self.array[0]

    def set_freqs(self, value):
        """Frequencies"""
        self.array[0, :] = value

    def amplitudes_rates(self):
        """Amplitudes rates"""
        return self.array[1]

    def amplitudes_desired(self):
        """Amplitudes desired"""
        return self.array[2]

    def set_amplitudes_desired(self, value):
        """Amplitudes desired"""
        self.array[2, :] = value


cdef class ConnectivityArrayCy(NetworkArray2D):
    """Connectivity array"""

    @classmethod
    def from_parameters(cls, connections, weights, desired_phases):
        """From each parameter"""
        return cls(np.stack([connections, weights, desired_phases], axis=1))

    def connections(self):
        """Connections"""
        return self.array[:][0, 1]

    def weights(self):
        """Weights"""
        return self.array[:][2]

    def desired_phases(self):
        """Weights"""
        return self.array[:][3]


cdef class JointsArrayCy(NetworkArray2D):
    """Oscillator array"""

    @classmethod
    def from_parameters(cls, offsets, rates):
        """From each parameter"""
        return cls(np.array([offsets, rates]))

    def offsets(self):
        """Joints angles offsets"""
        return self.array[0]

    def rates(self):
        """Joints angles offsets rates"""
        return self.array[1]

    def set_body_offset(self, value, n_body_joints=11):
        """Body offset"""
        self.array[0, :n_body_joints] = value

    def set_legs_offset(self, value, n_body_joints=11):
        """Legs offset"""
        self.array[0, n_body_joints:] = value


cdef class SensorsDataCy:
    """SensorsData"""

    def __init__(
            self,
            ContactsArrayCy contacts=None,
            ProprioceptionArrayCy proprioception=None,
            GpsArrayCy gps=None,
            HydrodynamicsArrayCy hydrodynamics=None
    ):
        super(SensorsDataCy, self).__init__()
        self.contacts = contacts
        self.proprioception = proprioception
        self.gps = gps
        self.hydrodynamics = hydrodynamics


cdef class ContactsArrayCy(NetworkArray3D):
    """Sensor array"""

    @classmethod
    def from_parameters(cls, n_iterations, n_contacts):
        """From parameters"""
        return cls(np.zeros([n_iterations, n_contacts, 9]))

    cpdef CTYPEv1 reaction(self, unsigned int iteration, unsigned int sensor_i):
        """Reaction force"""
        return self.array[iteration, sensor_i, 0:3]

    cpdef CTYPEv2 reaction_all(self, unsigned int sensor_i):
        """Reaction force"""
        return self.array[:, sensor_i, 0:3]

    cpdef CTYPEv1 friction(self, unsigned int iteration, unsigned int sensor_i):
        """Friction force"""
        return self.array[iteration, sensor_i, 3:6]

    cpdef CTYPEv2 friction_all(self, unsigned int sensor_i):
        """Friction force"""
        return self.array[:, sensor_i, 3:6]

    cpdef CTYPEv1 total(self, unsigned int iteration, unsigned int sensor_i):
        """Total force"""
        return self.array[iteration, sensor_i, 6:9]

    cpdef CTYPEv2 total_all(self, unsigned int sensor_i):
        """Total force"""
        return self.array[:, sensor_i, 6:9]


cdef class ProprioceptionArrayCy(NetworkArray3D):
    """Proprioception array"""

    @classmethod
    def from_parameters(cls, n_iterations, n_joints):
        """From parameters"""
        return cls(np.zeros([n_iterations, n_joints, 12]))

    cpdef CTYPE position(self, unsigned int iteration, unsigned int joint_i):
        """Joint position"""
        return self.array[iteration, joint_i, 0]

    cpdef CTYPEv1 positions(self, unsigned int iteration):
        """Joints positions"""
        return self.array[iteration, :, 0]

    cpdef CTYPEv2 positions_all(self):
        """Joints positions"""
        return self.array[:, :, 0]

    cpdef CTYPE velocity(self, unsigned int iteration, unsigned int joint_i):
        """Joint velocity"""
        return self.array[iteration, joint_i, 1]

    cpdef CTYPEv1 velocities(self, unsigned int iteration):
        """Joints velocities"""
        return self.array[iteration, :, 1]

    cpdef CTYPEv2 velocities_all(self):
        """Joints velocities"""
        return self.array[:, :, 1]

    cpdef CTYPEv1 force(self, unsigned int iteration, unsigned int joint_i):
        """Joint force"""
        return self.array[iteration, joint_i, 2:5]

    cpdef CTYPEv3 forces_all(self):
        """Joints forces"""
        return self.array[:, :, 2:5]

    cpdef CTYPEv1 torque(self, unsigned int iteration, unsigned int joint_i):
        """Joint torque"""
        return self.array[iteration, joint_i, 5:8]

    cpdef CTYPEv3 torques_all(self):
        """Joints torques"""
        return self.array[:, :, 5:8]

    cpdef CTYPE motor_torque(self, unsigned int iteration, unsigned int joint_i):
        """Joint velocity"""
        return self.array[iteration, joint_i, 8]

    cpdef CTYPEv2 motor_torques(self):
        """Joint velocity"""
        return self.array[:, :, 8]

    cpdef CTYPE active(self, unsigned int iteration, unsigned int joint_i):
        """Active torque"""
        return self.array[iteration, joint_i, 9]

    cpdef CTYPEv2 active_torques(self):
        """Active torques"""
        return self.array[:, :, 9]

    cpdef CTYPE spring(self, unsigned int iteration, unsigned int joint_i):
        """Passive spring torque"""
        return self.array[iteration, joint_i, 10]

    cpdef CTYPEv2 spring_torques(self):
        """Spring torques"""
        return self.array[:, :, 10]

    cpdef CTYPE damping(self, unsigned int iteration, unsigned int joint_i):
        """passive damping torque"""
        return self.array[iteration, joint_i, 11]

    cpdef CTYPEv2 damping_torques(self):
        """Damping torques"""
        return self.array[:, :, 11]


cdef class GpsArrayCy(NetworkArray3D):
    """Gps array"""

    @classmethod
    def from_parameters(cls, n_iterations, n_links):
        """From parameters"""
        return cls(np.zeros([n_iterations, n_links, 20]))

    cpdef CTYPEv1 com_position(self, unsigned int iteration, unsigned int link_i):
        """CoM position of a link"""
        return self.array[iteration, link_i, 0:3]

    cpdef CTYPEv1 com_orientation(self, unsigned int iteration, unsigned int link_i):
        """CoM orientation of a link"""
        return self.array[iteration, link_i, 3:7]

    cpdef CTYPEv1 urdf_position(self, unsigned int iteration, unsigned int link_i):
        """URDF position of a link"""
        return self.array[iteration, link_i, 7:10]

    cpdef CTYPEv3 urdf_positions(self):
        """URDF position of a link"""
        return self.array[:, :, 7:10]

    cpdef CTYPEv1 urdf_orientation(self, unsigned int iteration, unsigned int link_i):
        """URDF orientation of a link"""
        return self.array[iteration, link_i, 10:14]

    cpdef CTYPEv1 com_lin_velocity(self, unsigned int iteration, unsigned int link_i):
        """CoM linear velocity of a link"""
        return self.array[iteration, link_i, 14:17]

    cpdef CTYPEv3 com_lin_velocities(self):
        """CoM linear velocities"""
        return self.array[:, :, 14:17]

    cpdef CTYPEv1 com_ang_velocity(self, unsigned int iteration, unsigned int link_i):
        """CoM angular velocity of a link"""
        return self.array[iteration, link_i, 17:20]


cdef class HydrodynamicsArrayCy(NetworkArray3D):
    """Hydrodynamics array"""

    @classmethod
    def from_parameters(cls, n_iterations, n_links):
        """From parameters"""
        return cls(np.zeros([n_iterations, n_links, 6]))

    cpdef CTYPEv3 forces(self):
        """Forces"""
        return self.array[:, :, 0:3]

    cpdef CTYPEv3 torques(self):
        """Torques"""
        return self.array[:, :, 3:6]
