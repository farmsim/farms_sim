"""Position muscle model"""

cimport numpy as np
import numpy as np
from .muscle_cy cimport JointsMusclesCy


cdef class PositionMuscleCy(JointsMusclesCy):
    """Position muscle model"""
    cpdef void step(self, unsigned int iteration)
