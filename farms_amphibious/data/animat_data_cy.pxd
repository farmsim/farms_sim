"""Animat data"""

include 'types.pxd'
from farms_bullet.data.data_cy cimport SensorsDataCy
from farms_bullet.data.array cimport (
    DoubleArray1D,
    DoubleArray2D,
    IntegerArray2D,
)

cpdef enum ConnectionTypeJoint2Osc:
    POS2FREQ=0
    VEL2FREQ=1
    TOR2FREQ=2
    POS2AMP=3
    VEL2AMP=4
    TOR2AMP=5


cpdef enum ConnectionTypeContact2Osc:
    REACTION2FREQ=0
    REACTION2AMP=1
    FRICTION2FREQ=2
    FRICTION2AMP=3


cpdef enum ConnectionTypeHydro2Osc:
    LATERAL2FREQ=0
    LATERAL2AMP=1


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
    cdef public OscillatorsConnectivityCy osc_connectivity
    cdef public ConnectivityCy drive_connectivity
    cdef public JointsConnectivityCy joints_connectivity
    cdef public ContactsConnectivityCy contacts_connectivity
    cdef public HydroConnectivityCy hydro_connectivity


cdef class OscillatorNetworkStateCy(DoubleArray2D):
    """Network state"""
    cdef public unsigned int n_oscillators


cdef class DriveArrayCy(DoubleArray2D):
    """Drive array"""

    cdef inline DTYPE c_speed(self, unsigned int iteration) nogil:
        """Value"""
        return self.array[iteration, 0]

    cdef inline DTYPE c_turn(self, unsigned int iteration) nogil:
        """Value"""
        return self.array[iteration, 1]


cdef class DriveDependentArrayCy(DoubleArray2D):
    """Drive dependent array"""

    cdef public DTYPE value(self, unsigned int index, DTYPE drive)

    cdef inline unsigned int c_n_nodes(self) nogil:
        """Number of nodes"""
        return self.array.shape[0]

    cdef inline DTYPE c_gain(self, unsigned int index) nogil:
        """Gain"""
        return self.array[index, 0]

    cdef inline DTYPE c_bias(self, unsigned int index) nogil:
        """Bias"""
        return self.array[index, 1]

    cdef inline DTYPE c_low(self, unsigned int index) nogil:
        """Low"""
        return self.array[index, 2]

    cdef inline DTYPE c_high(self, unsigned int index) nogil:
        """High"""
        return self.array[index, 3]

    cdef inline DTYPE c_saturation(self, unsigned int index) nogil:
        """Saturation"""
        return self.array[index, 4]

    cdef inline DTYPE c_value(self, unsigned int index, DTYPE drive) nogil:
        """Value"""
        return (
            (self.c_gain(index)*drive + self.c_bias(index))
            if self.c_low(index) <= drive <= self.c_high(index)
            else self.c_saturation(index)
        )

    cdef inline DTYPE c_value_mod(self, unsigned int index, DTYPE drive1, DTYPE drive2) nogil:
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
    cdef public DoubleArray1D rates

    cdef inline unsigned int c_n_oscillators(self) nogil:
        """Number of oscillators"""
        return self.rates.array.shape[0]

    cdef inline DTYPE c_angular_frequency(self, unsigned int index, DTYPE drive) nogil:
        """Angular frequency"""
        return self.intrinsic_frequencies.c_value(index, drive)

    cdef inline DTYPE c_nominal_amplitude(self, unsigned int index, DTYPE drive) nogil:
        """Nominal amplitude"""
        return self.nominal_amplitudes.c_value(index, drive)

    cdef inline DTYPE c_rate(self, unsigned int index) nogil:
        """Rate"""
        return self.rates.array[index]


cdef class ConnectivityCy:
    """Connectivity array"""

    cdef readonly IntegerArray2D connections

    cpdef UITYPE input(self, unsigned int connection_i)
    cpdef UITYPE output(self, unsigned int connection_i)

    cdef inline UITYPE c_n_connections(self) nogil:
        """Number of connections"""
        return self.connections.array.shape[0]


cdef class OscillatorsConnectivityCy(ConnectivityCy):
    """oscillator connectivity array"""

    cdef readonly DoubleArray1D weights
    cdef readonly DoubleArray1D desired_phases

    cdef inline DTYPE c_weight(self, unsigned int iteration) nogil:
        """Weight"""
        return self.weights.array[iteration]

    cdef inline DTYPE c_desired_phase(self, unsigned int iteration) nogil:
        """Desired phase"""
        return self.desired_phases.array[iteration]


cdef class JointsConnectivityCy(ConnectivityCy):
    """Joint connectivity array"""

    cdef readonly DoubleArray1D weights

    cdef inline DTYPE c_weight(self, unsigned int iteration) nogil:
        """Weight"""
        return self.weights.array[iteration]


cdef class ContactsConnectivityCy(ConnectivityCy):
    """Contact connectivity array"""

    cdef readonly DoubleArray1D weights

    cdef inline DTYPE c_weight(self, unsigned int iteration) nogil:
        """Weight"""
        return self.weights.array[iteration]


cdef class HydroConnectivityCy(ConnectivityCy):
    """Hydrodynamics connectivity array"""

    cdef readonly DoubleArray1D amplitude
    cdef readonly DoubleArray1D frequency

    cdef inline DTYPE c_weight_frequency(self, unsigned int iteration) nogil:
        """Weight for hydrodynamics frequency"""
        return self.frequency.array[iteration]

    cdef inline DTYPE c_weight_amplitude(self, unsigned int iteration) nogil:
        """Weight for hydrodynamics amplitude"""
        return self.amplitude.array[iteration]


cdef class JointsArrayCy(DriveDependentArrayCy):
    """Drive dependent joints"""

    cdef inline unsigned int c_n_joints(self) nogil:
        """Number of joints"""
        return self.c_n_nodes()

    cdef inline DTYPE c_offset_desired(self, unsigned int index, DTYPE drive1, DTYPE drive2) nogil:
        """Desired offset"""
        return self.c_value_mod(index, drive1, drive2)

    cdef inline DTYPE c_rate(self, unsigned int index) nogil:
        """Rate"""
        return self.array[index, 5]
