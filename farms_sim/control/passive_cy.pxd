"""Passive muscle model"""

include 'types.pxd'
cimport numpy as np
import numpy as np
from .muscle_cy cimport JointsControlCy


cdef class PassiveJointCy(JointsControlCy):
    """Passive muscle model"""

    cdef public DTYPEv1 stiffness_coefficients
    cdef public DTYPEv1 damping_coefficients
    cdef public DTYPEv1 friction_coefficients

    cpdef void step(self, unsigned int iteration)
    cpdef np.ndarray stiffness(self, unsigned int iteration)
    cpdef np.ndarray damping(self, unsigned int iteration)
    cpdef np.ndarray friction(self, unsigned int iteration)
