"""Ekeberg muscle model"""

cimport numpy as np
import numpy as np
from .muscle_cy cimport JointsMusclesCy


cdef class EkebergMuscleCy(JointsMusclesCy):
    """Ekeberg muscle model"""
    cpdef void step(self, unsigned int iteration)
