"""Ekeberg muscle model"""

cimport numpy as np
import numpy as np
from .muscle_cy cimport MuscleCy


cdef class EkebergMuscleCy(MuscleCy):
    """Ekeberg muscle model"""
    cpdef void step(self, unsigned int iteration)
