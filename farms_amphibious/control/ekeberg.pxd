"""Ekeberg muscle model"""

include 'types.pxd'

cimport numpy as np
import numpy as np
from farms_data.sensors.data_cy cimport JointSensorArrayCy
from .muscle_cy cimport MuscleCy


cdef class EkebergMuscleCy(MuscleCy):
    """Ekeberg muscle model"""

    cpdef void step(self, unsigned int iteration)
