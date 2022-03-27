"""Muscle model"""

include 'sensor_convention.pxd'
cimport numpy as np
import numpy as np


cdef np.ndarray get_joints_data(
    unsigned int iteration,
    JointSensorArrayCy joints_data,
    unsigned int n_joints,
    UITYPEv1 indices,
    unsigned int array_index,
):
    """Get joints data"""
    cdef unsigned int joint_i
    cdef np.ndarray data = np.zeros(n_joints, dtype=np.double)
    for joint_i in range(n_joints):
        data[joint_i] = joints_data.array[iteration, indices[joint_i], array_index]
    return data


cdef void set_joints_data(
    unsigned int iteration,
    JointSensorArrayCy joints_data,
    unsigned int n_joints,
    UITYPEv1 indices,
    unsigned int array_index,
    DTYPEv1 data,
) nogil:
    """Set joints data"""
    cdef unsigned int joint_i
    for joint_i in range(n_joints):
        joints_data.array[iteration, indices[joint_i], array_index] = data[joint_i]


cdef class JointsControlCy:
    """Joints control"""

    def __init__(
            self,
            list joints_names,
            JointSensorArrayCy joints_data,
            UITYPEv1 indices,
            DTYPEv1 gain,
            DTYPEv1 bias,
    ):
        super().__init__()
        self.joints_names = joints_names
        self.joints_data = joints_data
        self.indices = indices
        self.n_joints = len(indices)
        self.transform_gain = gain
        self.transform_bias = bias

    cpdef np.ndarray position_cmds(self, unsigned int iteration):
        """Positions"""
        return get_joints_data(
            iteration=iteration,
            joints_data=self.joints_data,
            n_joints=self.n_joints,
            indices=self.indices,
            array_index=JOINT_CMD_POSITION,
        )

    cpdef np.ndarray velocity_cmds(self, unsigned int iteration):
        """Velocities"""
        return get_joints_data(
            iteration=iteration,
            joints_data=self.joints_data,
            n_joints=self.n_joints,
            indices=self.indices,
            array_index=JOINT_CMD_VELOCITY,
        )

    cpdef np.ndarray torque_cmds(self, unsigned int iteration):
        """Torques"""
        return get_joints_data(
            iteration=iteration,
            joints_data=self.joints_data,
            n_joints=self.n_joints,
            indices=self.indices,
            array_index=JOINT_CMD_TORQUE,
        )

    cpdef void set_position_cmds(
        self,
        unsigned int iteration,
        DTYPEv1 data,
    ):
        """Set position commands"""
        set_joints_data(
            iteration=iteration,
            joints_data=self.joints_data,
            n_joints=self.n_joints,
            indices=self.indices,
            array_index=JOINT_CMD_POSITION,
            data=data,
        )

    cpdef void set_velocity_cmds(
        self,
        unsigned int iteration,
        DTYPEv1 data,
    ):
        """Set velocity commands"""
        set_joints_data(
            iteration=iteration,
            joints_data=self.joints_data,
            n_joints=self.n_joints,
            indices=self.indices,
            array_index=JOINT_CMD_VELOCITY,
            data=data,
        )

    cpdef void set_torque_cmds(
        self,
        unsigned int iteration,
        DTYPEv1 data,
    ):
        """Set torque commands"""
        set_joints_data(
            iteration=iteration,
            joints_data=self.joints_data,
            n_joints=self.n_joints,
            indices=self.indices,
            array_index=JOINT_CMD_TORQUE,
            data=data,
        )
