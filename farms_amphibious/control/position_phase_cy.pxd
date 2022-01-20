"""Position phase model"""

include 'types.pxd'

cimport numpy as np
import numpy as np
from farms_data.sensors.data_cy cimport JointSensorArrayCy
from .joints_control_cy cimport JointsControlCy
from .network_cy cimport NetworkCy


cdef class PositionPhaseCy(JointsControlCy):
    """Position phase model"""

    cdef public NetworkCy network
    cdef public UITYPEv2 osc_indices

    cpdef void step(self, unsigned int iteration)
