"""Ekeberg muscle model"""

include 'types.pxd'

cimport numpy as np
import numpy as np
from farms_data.sensors.data_cy cimport JointSensorArrayCy
from .network_cy cimport NetworkCy


cdef class EkebergMuscleCy:
    """Ekeberg muscle model"""

    cdef public NetworkCy network
    cdef public JointSensorArrayCy joints
    cdef public unsigned int n_muscles
    cdef public UITYPEv1 indices
    cdef public DTYPEv2 parameters
    cdef public UITYPEv2 groups
    cdef public DTYPEv1 transform_gain
    cdef public DTYPEv1 transform_bias

    cpdef void step(self, unsigned int iteration)
    cpdef np.ndarray torques(self, unsigned int iteration)
    cpdef np.ndarray torques_implicit(self, unsigned int iteration)
    cpdef np.ndarray damping(self, unsigned int iteration)
    cpdef void log_active(self, unsigned int iteration, DTYPEv1 torques)
    cpdef void log_passive_stiffness(self, unsigned int iteration, DTYPEv1 torques)
    cpdef void log_damping(self, unsigned int iteration, DTYPEv1 torques)
