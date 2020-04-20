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

    cpdef INDEX input(self, connection_i)
    cpdef INDEX output(self, connection_i)

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

    cpdef CTYPEv1 reaction(self, unsigned int iteration, unsigned int sensor_i)
    cpdef CTYPEv2 reaction_all(self, unsigned int sensor_i)
    cpdef CTYPEv1 friction(self, unsigned int iteration, unsigned int sensor_i)
    cpdef CTYPEv2 friction_all(self, unsigned int sensor_i)
    cpdef CTYPEv1 total(self, unsigned int iteration, unsigned int sensor_i)
    cpdef CTYPEv2 total_all(self, unsigned int sensor_i)

    cdef inline CTYPE c_force_x(self, unsigned iteration, unsigned int index) nogil:
        """Force z"""
        return self.array[iteration, index, 0]

    cdef inline CTYPE c_force_y(self, unsigned iteration, unsigned int index) nogil:
        """Force z"""
        return self.array[iteration, index, 1]

    cdef inline CTYPE c_force_z(self, unsigned iteration, unsigned int index) nogil:
        """Force z"""
        return self.array[iteration, index, 2]


cdef class ProprioceptionArrayCy(NetworkArray3D):
    """Proprioception array"""

    cpdef CTYPE position(self, unsigned int iteration, unsigned int joint_i)
    cpdef CTYPEv1 positions(self, unsigned int iteration)
    cpdef CTYPEv2 positions_all(self)
    cpdef CTYPE velocity(self, unsigned int iteration, unsigned int joint_i)
    cpdef CTYPEv1 velocities(self, unsigned int iteration)
    cpdef CTYPEv2 velocities_all(self)
    cpdef CTYPEv1 force(self, unsigned int iteration, unsigned int joint_i)
    cpdef CTYPEv3 forces_all(self)
    cpdef CTYPEv1 torque(self, unsigned int iteration, unsigned int joint_i)
    cpdef CTYPEv3 torques_all(self)
    cpdef CTYPE motor_torque(self, unsigned int iteration, unsigned int joint_i)
    cpdef CTYPEv2 motor_torques(self)
    cpdef CTYPE active(self, unsigned int iteration, unsigned int joint_i)
    cpdef CTYPEv2 active_torques(self)
    cpdef CTYPE spring(self, unsigned int iteration, unsigned int joint_i)
    cpdef CTYPEv2 spring_torques(self)
    cpdef CTYPE damping(self, unsigned int iteration, unsigned int joint_i)
    cpdef CTYPEv2 damping_torques(self)


cdef class GpsArrayCy(NetworkArray3D):
    """Gps array"""

    cpdef public CTYPEv1 com_position(self, unsigned int iteration, unsigned int link_i)
    cpdef public CTYPEv1 com_orientation(self, unsigned int iteration, unsigned int link_i)
    cpdef public CTYPEv1 urdf_position(self, unsigned int iteration, unsigned int link_i)
    cpdef public CTYPEv3 urdf_positions(self)
    cpdef public CTYPEv1 urdf_orientation(self, unsigned int iteration, unsigned int link_i)
    cpdef public CTYPEv1 com_lin_velocity(self, unsigned int iteration, unsigned int link_i)
    cpdef public CTYPEv3 com_lin_velocities(self)
    cpdef public CTYPEv1 com_ang_velocity(self, unsigned int iteration, unsigned int link_i)


cdef class HydrodynamicsArrayCy(NetworkArray3D):
    """Hydrodynamics array"""

    cpdef public CTYPEv3 forces(self)
    cpdef public CTYPEv3 torques(self)

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
