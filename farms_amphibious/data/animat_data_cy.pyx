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
            drives,
            oscillators,
            osc_connectivity,
            contacts_connectivity,
            hydro_connectivity
    ):
        super(NetworkParametersCy, self).__init__()
        self.drives = drives
        self.oscillators = oscillators
        self.osc_connectivity = osc_connectivity
        self.contacts_connectivity = contacts_connectivity
        self.hydro_connectivity = hydro_connectivity


cdef class OscillatorNetworkStateCy(NetworkArray2D):
    """Network state"""

    def __init__(self, state, n_oscillators):
        super(OscillatorNetworkStateCy, self).__init__(state)
        self.n_oscillators = n_oscillators


cdef class DriveDependentArrayCy(NetworkArray2D):
    """Drive dependent array"""

    @classmethod
    def from_parameters(cls, gain, bias, low, high, saturation):
        """From each parameter"""
        return cls(np.array([gain, bias, low, high, saturation]))

    cdef DTYPE value(self, unsigned int index, DTYPE drive):
        """Value for a given drive"""
        return (
            self.gain[index]*drive + self.bias[index]
            if self.low[index] <= drive <= self.high[index]
            else self.saturation[index]
        )


cdef class OscillatorsCy:
    """Oscillator array"""

    def __init__(self, intrinsic_frequencies, nominal_amplitudes, rates):
        super(OscillatorsCy, self).__init__()
        self.intrinsic_frequencies = DriveDependentArrayCy(intrinsic_frequencies)
        self.nominal_amplitudes = DriveDependentArrayCy(nominal_amplitudes)
        self.rates = NetworkArray1D(rates)


cdef class ConnectivityCy:
    """Connectivity array"""

    def __init__(self, connections):
        super(ConnectivityCy, self).__init__()
        if connections is not None and list(connections):
            assert np.shape(connections)[1] == 2, (
                'Connections should be of dim 2, got {}'.format(
                    np.shape(connections)[1]
                )
            )
            self.connections = IntegerArray2D(connections)
        else:
            self.connections = IntegerArray2D(None)

    cpdef INDEX input(self, unsigned int connection_i):
        """Node input"""
        self.array[connection_i, 0]

    cpdef INDEX output(self, unsigned int connection_i):
        """Node input"""
        self.array[connection_i, 1]


cdef class OscillatorConnectivityCy(ConnectivityCy):
    """Oscillator connectivity array"""

    def __init__(self, connections, weights, desired_phases):
        super(OscillatorConnectivityCy, self).__init__(connections)
        if connections is not None and list(connections):
            size = np.shape(connections)[0]
            assert size == len(weights), (
                'Size of connections {} != size of size of weights {}'.format(
                    size,
                    len(weights),
                )
            )
            assert size == len(desired_phases), (
                'Size of connections {} != size of size of phases {}'.format(
                    size,
                    len(desired_phases),
                )
            )
            self.weights = NetworkArray1D(weights)
            self.desired_phases = NetworkArray1D(desired_phases)
        else:
            self.weights = NetworkArray1D(None)
            self.desired_phases = NetworkArray1D(None)


cdef class ContactConnectivityCy(ConnectivityCy):
    """Contact connectivity array"""

    def __init__(self, connections, weights):
        super(ContactConnectivityCy, self).__init__(connections)
        if connections is not None and list(connections):
            size = np.shape(connections)[0]
            assert size == len(weights), (
                'Size of connections {} != size of size of weights {}'.format(
                    size,
                    len(weights),
                )
            )
            self.weights = NetworkArray1D(weights)
        else:
            self.weights = NetworkArray1D(None)


cdef class HydroConnectivityCy(ConnectivityCy):
    """Connectivity array"""

    def __init__(self, connections, frequency, amplitude):
        super(HydroConnectivityCy, self).__init__(connections)
        if connections is not None and list(connections):
            size = np.shape(connections)[0]
            assert size == len(frequency), (
                'Size of connections {} != size of size of frequency {}'.format(
                    size,
                    len(frequency),
                )
            )
            assert size == len(amplitude), (
                'Size of connections {} != size of size of amplitude {}'.format(
                    size,
                    len(amplitude),
                )
            )
            self.frequency = NetworkArray1D(frequency)
            self.amplitude = NetworkArray1D(amplitude)
        else:
            self.frequency = NetworkArray1D(None)
            self.amplitude = NetworkArray1D(None)


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
