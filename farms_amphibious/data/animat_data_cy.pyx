"""Animat data"""

import numpy as np
cimport numpy as np


cdef class AnimatDataCy:
    """Network parameter"""
    pass


cdef class NetworkParametersCy:
    """Network parameter"""
    pass


cdef class DriveDependentArrayCy(DoubleArray2D):
    """Drive dependent array"""

    cdef DTYPE value(self, unsigned int index, DTYPE drive):
        """Value for a given drive"""
        return (
            self.gain[index]*drive + self.bias[index]
            if self.low[index] <= drive <= self.high[index]
            else self.saturation[index]
        )


cdef class OscillatorsCy:
    """Oscillator array"""
    pass


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

    cpdef UITYPE input(self, unsigned int connection_i):
        """Node input"""
        self.array[connection_i, 0]

    cpdef UITYPE output(self, unsigned int connection_i):
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
            self.weights = DoubleArray1D(weights)
            self.desired_phases = DoubleArray1D(desired_phases)
        else:
            self.weights = DoubleArray1D(None)
            self.desired_phases = DoubleArray1D(None)


cdef class JointConnectivityCy(ConnectivityCy):
    """Joint connectivity array"""

    def __init__(self, connections, weights):
        super(JointConnectivityCy, self).__init__(connections)
        if connections is not None and list(connections):
            size = np.shape(connections)[0]
            assert size == len(weights), (
                'Size of connections {} != size of size of weights {}'.format(
                    size,
                    len(weights),
                )
            )
            self.weights = DoubleArray1D(weights)
        else:
            self.weights = DoubleArray1D(None)


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
            self.weights = DoubleArray1D(weights)
        else:
            self.weights = DoubleArray1D(None)


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
            self.frequency = DoubleArray1D(frequency)
            self.amplitude = DoubleArray1D(amplitude)
        else:
            self.frequency = DoubleArray1D(None)
            self.amplitude = DoubleArray1D(None)
