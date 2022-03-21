"""Position phase model"""

include 'types.pxd'

cimport numpy as np
import numpy as np
from farms_data.sensors.data_cy cimport JointSensorArrayCy
from farms_data.amphibious.animat_data_cy cimport OscillatorNetworkStateCy
from .joints_control_cy cimport JointsControlCy


cdef class PositionPhaseCy(JointsControlCy):
    """Position phase model"""

    cdef public OscillatorNetworkStateCy state
    cdef public UITYPEv2 osc_indices
    cdef public double weight
    cdef public double offset
    cdef public double threshold

    cpdef void step(self, unsigned int iteration)
