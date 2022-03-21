"""Joints control"""

include 'types.pxd'

cimport numpy as np
import numpy as np
from farms_data.sensors.data_cy cimport JointSensorArrayCy


cdef np.ndarray get_joints_data(
    unsigned int iteration,
    JointSensorArrayCy joints_data,
    unsigned int n_joints,
    UITYPEv1 indices,
    unsigned int array_index,
)


cdef void set_joints_data(
    unsigned int iteration,
    JointSensorArrayCy joints_data,
    unsigned int n_joints,
    UITYPEv1 indices,
    unsigned int array_index,
    DTYPEv1 data,
) nogil


cdef class JointsControlCy:
    """Joints control"""

    cdef public list joints_names
    cdef public JointSensorArrayCy joints_data
    cdef public unsigned int n_joints
    cdef public UITYPEv1 indices
    cdef public DTYPEv1 transform_gain
    cdef public DTYPEv1 transform_bias

    cpdef np.ndarray position_cmds(self, unsigned int iteration)
    cpdef np.ndarray velocity_cmds(self, unsigned int iteration)
    cpdef np.ndarray torque_cmds(self, unsigned int iteration)
    cpdef void set_position_cmds(self, unsigned int iteration, DTYPEv1 torques)
    cpdef void set_velocity_cmds(self, unsigned int iteration, DTYPEv1 torques)
    cpdef void set_torque_cmds(self, unsigned int iteration, DTYPEv1 torques)
