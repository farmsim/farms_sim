"""Network"""

include 'types.pxd'
include 'sensor_convention.pxd'

cimport numpy as np
import numpy as np


cdef class NetworkCy:
    """Network Cython"""
    cdef public DTYPEv2 state_array
    cdef public DTYPEv2 drives_array
    cdef public unsigned int n_iterations
    cdef public unsigned int n_oscillators
    cdef public DTYPEv1 dstate
    cpdef DTYPEv1 phases(self, unsigned int iteration)
    cpdef DTYPEv2 phases_all(self)
    cpdef DTYPEv1 amplitudes(self, unsigned int iteration)
    cpdef DTYPEv2 amplitudes_all(self)
    cpdef DTYPEv1 offsets(self, unsigned int iteration)
    cpdef DTYPEv2 offsets_all(self)
    cpdef np.ndarray outputs(self, unsigned int iteration)
