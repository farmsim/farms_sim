"""Animat data"""

include 'types.pxd'
from .array cimport (
    NetworkArray1D,
    NetworkArray2D,
    NetworkArray3D,
    IntegerArray2D,
)


cdef class AnimatDataCy:
    """Network parameter"""
    cdef public OscillatorNetworkStateCy state
    cdef public NetworkParametersCy network
    cdef public JointsArrayCy joints
    cdef public SensorsDataCy sensors


cdef class NetworkParametersCy:
    """Network parameter"""
    cdef public DriveArrayCy drives
    cdef public OscillatorsCy oscillators
    cdef public OscillatorConnectivityCy osc_connectivity
    cdef public ContactConnectivityCy contacts_connectivity
    cdef public HydroConnectivityCy hydro_connectivity


cdef class OscillatorNetworkStateCy(NetworkArray2D):
    """Network state"""
    cdef public unsigned int n_oscillators
    cdef public unsigned int _iterations


cdef class DriveArrayCy(NetworkArray2D):
    """Drive array"""

    cdef inline CTYPE c_speed(self, unsigned int iteration) nogil:
        """Value"""
        return self.array[iteration, 0]

    cdef inline CTYPE c_turn(self, unsigned int iteration) nogil:
        """Value"""
        return self.array[iteration, 1]


cdef class DriveDependentArrayCy(NetworkArray2D):
    """Drive dependent array"""

    cdef public CTYPE value(self, unsigned int index, CTYPE drive)

    cdef inline unsigned int c_n_nodes(self) nogil:
        """Number of nodes"""
        return self.array.shape[0]

    cdef inline CTYPE c_gain(self, unsigned int index) nogil:
        """Gain"""
        return self.array[index, 0]

    cdef inline CTYPE c_bias(self, unsigned int index) nogil:
        """Bias"""
        return self.array[index, 1]

    cdef inline CTYPE c_low(self, unsigned int index) nogil:
        """Low"""
        return self.array[index, 2]

    cdef inline CTYPE c_high(self, unsigned int index) nogil:
        """High"""
        return self.array[index, 3]

    cdef inline CTYPE c_saturation(self, unsigned int index) nogil:
        """Saturation"""
        return self.array[index, 4]

    cdef inline CTYPE c_value(self, unsigned int index, CTYPE drive) nogil:
        """Value"""
        return (
            (self.c_gain(index)*drive + self.c_bias(index))
            if self.c_low(index) <= drive <= self.c_high(index)
            else self.c_saturation(index)
        )

    cdef inline CTYPE c_value_mod(self, unsigned int index, CTYPE drive1, CTYPE drive2) nogil:
        """Value"""
        return (
            (self.c_gain(index)*drive1 + self.c_bias(index))
            if self.c_low(index) <= drive2 <= self.c_high(index)
            else self.c_saturation(index)
        )


cdef class OscillatorsCy:
    """Oscillator array"""
    cdef public DriveDependentArrayCy intrinsic_frequencies
    cdef public DriveDependentArrayCy nominal_amplitudes
    cdef public NetworkArray1D rates

    cdef inline unsigned int c_n_oscillators(self) nogil:
        """Number of oscillators"""
        return self.rates.array.shape[0]

    cdef inline CTYPE c_angular_frequency(self, unsigned int index, CTYPE drive) nogil:
        """Angular frequency"""
        return self.intrinsic_frequencies.c_value(index, drive)

    cdef inline CTYPE c_nominal_amplitude(self, unsigned int index, CTYPE drive) nogil:
        """Nominal amplitude"""
        return self.nominal_amplitudes.c_value(index, drive)

    cdef inline CTYPE c_rate(self, unsigned int index) nogil:
        """Rate"""
        return self.rates.array[index]


cdef class ConnectivityCy:
    """Connectivity array"""

    cdef readonly IntegerArray2D connections

    cpdef INDEX input(self, unsigned int connection_i)
    cpdef INDEX output(self, unsigned int connection_i)

    cdef inline INDEX c_n_connections(self) nogil:
        """Number of connections"""
        return self.connections.array.shape[0]


cdef class OscillatorConnectivityCy(ConnectivityCy):
    """oscillator connectivity array"""

    cdef readonly NetworkArray1D weights
    cdef readonly NetworkArray1D desired_phases

    cdef inline CTYPE c_weight(self, unsigned int iteration) nogil:
        """Weight"""
        return self.weights.array[iteration]

    cdef inline CTYPE c_desired_phase(self, unsigned int iteration) nogil:
        """Desired phase"""
        return self.desired_phases.array[iteration]


cdef class ContactConnectivityCy(ConnectivityCy):
    """Contact connectivity array"""

    cdef readonly NetworkArray1D weights

    cdef inline CTYPE c_weight(self, unsigned int iteration) nogil:
        """Weight"""
        return self.weights.array[iteration]


cdef class HydroConnectivityCy(ConnectivityCy):
    """Hydrodynamics connectivity array"""

    cdef readonly NetworkArray1D amplitude
    cdef readonly NetworkArray1D frequency

    cdef inline CTYPE c_weight_frequency(self, unsigned int iteration) nogil:
        """Weight for hydrodynamics frequency"""
        return self.frequency.array[iteration]

    cdef inline CTYPE c_weight_amplitude(self, unsigned int iteration) nogil:
        """Weight for hydrodynamics amplitude"""
        return self.amplitude.array[iteration]


cdef class JointsArrayCy(DriveDependentArrayCy):
    """Drive dependent joints"""

    cdef inline unsigned int c_n_joints(self) nogil:
        """Number of joints"""
        return self.c_n_nodes()

    cdef inline CTYPE c_offset_desired(self, unsigned int index, CTYPE drive1, CTYPE drive2) nogil:
        """Desired offset"""
        return self.c_value_mod(index, drive1, drive2)

    cdef inline CTYPE c_rate(self, unsigned int index) nogil:
        """Rate"""
        return self.array[index, 5]


cdef class SensorsDataCy:
    """SensorsData"""
    cdef public ContactsArrayCy contacts
    cdef public ProprioceptionArrayCy proprioception
    cdef public GpsArrayCy gps
    cdef public HydrodynamicsArrayCy hydrodynamics


cdef class ContactsArrayCy(NetworkArray3D):
    """Sensor array"""

    cdef inline CTYPEv1 c_all(self, unsigned iteration, unsigned int index) nogil:
        """Reaction"""
        return self.array[iteration, index, :]

    cdef inline CTYPEv1 c_reaction(self, unsigned iteration, unsigned int index) nogil:
        """Reaction"""
        return self.array[iteration, index, 0:3]

    cdef inline CTYPE c_reaction_x(self, unsigned iteration, unsigned int index) nogil:
        """Reaction x"""
        return self.array[iteration, index, 0]

    cdef inline CTYPE c_reaction_y(self, unsigned iteration, unsigned int index) nogil:
        """Reaction y"""
        return self.array[iteration, index, 1]

    cdef inline CTYPE c_reaction_z(self, unsigned iteration, unsigned int index) nogil:
        """Reaction z"""
        return self.array[iteration, index, 2]

    cdef inline CTYPEv1 c_friction(self, unsigned iteration, unsigned int index) nogil:
        """Friction"""
        return self.array[iteration, index, 3:6]

    cdef inline CTYPE c_friction_x(self, unsigned iteration, unsigned int index) nogil:
        """Friction x"""
        return self.array[iteration, index, 3]

    cdef inline CTYPE c_friction_y(self, unsigned iteration, unsigned int index) nogil:
        """Friction y"""
        return self.array[iteration, index, 4]

    cdef inline CTYPE c_friction_z(self, unsigned iteration, unsigned int index) nogil:
        """Friction z"""
        return self.array[iteration, index, 5]

    cdef inline CTYPEv1 c_total(self, unsigned iteration, unsigned int index) nogil:
        """Total"""
        return self.array[iteration, index, 6:9]

    cdef inline CTYPE c_total_x(self, unsigned iteration, unsigned int index) nogil:
        """Total x"""
        return self.array[iteration, index, 6]

    cdef inline CTYPE c_total_y(self, unsigned iteration, unsigned int index) nogil:
        """Total y"""
        return self.array[iteration, index, 7]

    cdef inline CTYPE c_total_z(self, unsigned iteration, unsigned int index) nogil:
        """Total z"""
        return self.array[iteration, index, 8]


cdef class ProprioceptionArrayCy(NetworkArray3D):
    """Proprioception array"""

    cdef inline CTYPE position_cy(self, unsigned int iteration, unsigned int joint_i):
        """Joint position"""
        return self.array[iteration, joint_i, 0]

    cdef inline CTYPEv1 positions_cy(self, unsigned int iteration):
        """Joints positions"""
        return self.array[iteration, :, 0]

    cdef inline CTYPEv2 positions_all_cy(self):
        """Joints positions"""
        return self.array[:, :, 0]

    cdef inline CTYPE velocity_cy(self, unsigned int iteration, unsigned int joint_i):
        """Joint velocity"""
        return self.array[iteration, joint_i, 1]

    cdef inline CTYPEv1 velocities_cy(self, unsigned int iteration):
        """Joints velocities"""
        return self.array[iteration, :, 1]

    cdef inline CTYPEv2 velocities_all_cy(self):
        """Joints velocities"""
        return self.array[:, :, 1]

    cdef inline CTYPEv1 force_cy(self, unsigned int iteration, unsigned int joint_i):
        """Joint force"""
        return self.array[iteration, joint_i, 2:5]

    cdef inline CTYPEv3 forces_all_cy(self):
        """Joints forces"""
        return self.array[:, :, 2:5]

    cdef inline CTYPEv1 torque_cy(self, unsigned int iteration, unsigned int joint_i):
        """Joint torque"""
        return self.array[iteration, joint_i, 5:8]

    cdef inline CTYPEv3 torques_all_cy(self):
        """Joints torques"""
        return self.array[:, :, 5:8]

    cdef inline CTYPE motor_torque_cy(self, unsigned int iteration, unsigned int joint_i):
        """Joint velocity"""
        return self.array[iteration, joint_i, 8]

    cdef inline CTYPEv2 motor_torques_cy(self):
        """Joint velocity"""
        return self.array[:, :, 8]

    cdef inline CTYPE active_cy(self, unsigned int iteration, unsigned int joint_i):
        """Active torque"""
        return self.array[iteration, joint_i, 9]

    cdef inline CTYPEv2 active_torques_cy(self):
        """Active torques"""
        return self.array[:, :, 9]

    cdef inline CTYPE spring_cy(self, unsigned int iteration, unsigned int joint_i):
        """Passive spring torque"""
        return self.array[iteration, joint_i, 10]

    cdef inline CTYPEv2 spring_torques_cy(self):
        """Spring torques"""
        return self.array[:, :, 10]

    cdef inline CTYPE damping_cy(self, unsigned int iteration, unsigned int joint_i):
        """passive damping torque"""
        return self.array[iteration, joint_i, 11]

    cdef inline CTYPEv2 damping_torques_cy(self):
        """Damping torques"""
        return self.array[:, :, 11]


cdef class GpsArrayCy(NetworkArray3D):
    """Gps array"""

    cdef inline CTYPEv1 com_position_cy(self, unsigned int iteration, unsigned int link_i):
        """CoM position of a link"""
        return self.array[iteration, link_i, 0:3]

    cdef inline CTYPEv1 com_orientation_cy(self, unsigned int iteration, unsigned int link_i):
        """CoM orientation of a link"""
        return self.array[iteration, link_i, 3:7]

    cdef inline CTYPEv1 urdf_position_cy(self, unsigned int iteration, unsigned int link_i):
        """URDF position of a link"""
        return self.array[iteration, link_i, 7:10]

    cdef inline CTYPEv3 urdf_positions_cy(self):
        """URDF position of a link"""
        return self.array[:, :, 7:10]

    cdef inline CTYPEv1 urdf_orientation_cy(self, unsigned int iteration, unsigned int link_i):
        """URDF orientation of a link"""
        return self.array[iteration, link_i, 10:14]

    cdef inline CTYPEv1 com_lin_velocity_cy(self, unsigned int iteration, unsigned int link_i):
        """CoM linear velocity of a link"""
        return self.array[iteration, link_i, 14:17]

    cdef inline CTYPEv3 com_lin_velocities_cy(self):
        """CoM linear velocities"""
        return self.array[:, :, 14:17]

    cdef inline CTYPEv1 com_ang_velocity_cy(self, unsigned int iteration, unsigned int link_i):
        """CoM angular velocity of a link"""
        return self.array[iteration, link_i, 17:20]


cdef class HydrodynamicsArrayCy(NetworkArray3D):
    """Hydrodynamics array"""

    cdef inline CTYPE c_force_x(self, unsigned iteration, unsigned int index) nogil:
        """Force x"""
        return self.array[iteration, index, 0]

    cdef inline CTYPE c_force_y(self, unsigned iteration, unsigned int index) nogil:
        """Force y"""
        return self.array[iteration, index, 1]

    cdef inline CTYPE c_force_z(self, unsigned iteration, unsigned int index) nogil:
        """Force z"""
        return self.array[iteration, index, 2]

    cdef inline CTYPE c_torque_x(self, unsigned iteration, unsigned int index) nogil:
        """Torque x"""
        return self.array[iteration, index, 0]

    cdef inline CTYPE c_torque_y(self, unsigned iteration, unsigned int index) nogil:
        """Torque y"""
        return self.array[iteration, index, 1]

    cdef inline CTYPE c_torque_z(self, unsigned iteration, unsigned int index) nogil:
        """Torque z"""
        return self.array[iteration, index, 2]
