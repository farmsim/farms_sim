"""Joints muscles"""

include 'types.pxd'

cimport numpy as np
import numpy as np
from farms_data.sensors.data_cy cimport JointSensorArrayCy
from farms_data.amphibious.animat_data_cy cimport OscillatorNetworkStateCy
from .joints_control_cy cimport JointsControlCy


cdef class JointsMusclesCy(JointsControlCy):
    """Joints muscles"""

    cdef public OscillatorNetworkStateCy state
    cdef public DTYPEv2 parameters
    cdef public UITYPEv2 osc_indices

    cpdef np.ndarray torques_implicit(self, unsigned int iteration)
    cpdef np.ndarray damping(self, unsigned int iteration)
    cpdef np.ndarray friction(self, unsigned int iteration)
    cpdef void set_active(self, unsigned int iteration, DTYPEv1 torques)
    cpdef void set_passive_stiffness(self, unsigned int iteration, DTYPEv1 torques)
    cpdef void set_damping(self, unsigned int iteration, DTYPEv1 torques)
    cpdef void set_friction(self, unsigned int iteration, DTYPEv1 torques)
