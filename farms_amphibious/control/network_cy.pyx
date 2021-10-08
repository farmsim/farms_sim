"""Network"""

import numpy as np


cdef class NetworkCy:
    """Network Cython"""

    def __init__(self, data, dstate):
        super().__init__()
        self.n_oscillators = data.state.n_oscillators
        self.state_array = data.state.array
        self.drives_array = data.network.drives.array
        self.dstate = dstate

    cpdef DTYPEv1 phases(self, unsigned int iteration):
        """Oscillators phases"""
        return self.state_array[iteration, :self.n_oscillators]

    cpdef DTYPEv2 phases_all(self):
        """Oscillators phases"""
        return self.state_array[:, :self.n_oscillators]

    cpdef DTYPEv1 amplitudes(self, unsigned int iteration):
        """Amplitudes"""
        return self.state_array[
            iteration,
            self.n_oscillators:2*self.n_oscillators
        ]

    cpdef DTYPEv2 amplitudes_all(self):
        """Amplitudes"""
        return self.state_array[:, self.n_oscillators:2*self.n_oscillators]

    cpdef DTYPEv1 offsets(self, unsigned int iteration):
        """Offset"""
        return self.state_array[iteration, 2*self.n_oscillators:]

    cpdef DTYPEv2 offsets_all(self):
        """Offset"""
        return self.state_array[:, 2*self.n_oscillators:]

    cpdef np.ndarray outputs(self, unsigned int iteration):
        """Outputs"""
        return self.amplitudes(iteration)*(1 + np.cos(self.phases(iteration)))
