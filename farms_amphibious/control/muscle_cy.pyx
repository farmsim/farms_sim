"""Muscle model"""

cimport numpy as np
import numpy as np


cdef void log_torque(
    unsigned int iteration,
    JointSensorArrayCy joints,
    UITYPEv1 indices,
    DTYPEv1 torques,
    unsigned int n_muscles,
    unsigned int log_index,
) nogil:
    """Log torque"""
    cdef unsigned int i
    for i in range(n_muscles):
        joints.array[iteration, indices[i], log_index] = torques[i]


cdef np.ndarray torques(
    unsigned int iteration,
    JointSensorArrayCy joints,
    unsigned int n_muscles,
    UITYPEv1 indices,
    unsigned int index,
):
    """Torques"""
    cdef unsigned int muscle_i, joint_i
    cdef np.ndarray torques = np.zeros(n_muscles, dtype=np.double)
    for muscle_i in range(n_muscles):
        joint_i = indices[muscle_i]
        torques[muscle_i] = joints.array[iteration, joint_i, index]
    return torques


cdef class MuscleCy:
    """Muscle model"""

    def __init__(
            self,
            NetworkCy network,
            JointSensorArrayCy joints,
            unsigned int n_muscles,
            UITYPEv1 indices,
            DTYPEv2 parameters,
            UITYPEv2 groups,
            DTYPEv1 gain,
            DTYPEv1 bias,
    ):
        super().__init__()
        self.network = network
        self.joints = joints
        self.n_muscles = n_muscles
        self.indices = indices
        self.parameters = parameters
        self.groups = groups
        self.transform_gain = gain
        self.transform_bias = bias

    cpdef np.ndarray torques(self, unsigned int iteration):
        """Torques"""
        return torques(
            iteration=iteration,
            joints=self.joints,
            n_muscles=self.n_muscles,
            indices=self.indices,
            index=8,
        )

    cpdef np.ndarray torques_implicit(self, unsigned int iteration):
        """Torques"""
        cdef unsigned int muscle_i, joint_i
        cdef np.ndarray torques = np.zeros(self.n_muscles, dtype=np.double)
        for muscle_i in range(self.n_muscles):
            joint_i = self.indices[muscle_i]
            torques[muscle_i] = (
                self.joints.array[iteration, joint_i, 9]
                + self.joints.array[iteration, joint_i, 10]
            )
        return torques

    cpdef np.ndarray damping(self, unsigned int iteration):
        """Torques"""
        return torques(
            iteration=iteration,
            joints=self.joints,
            n_muscles=self.n_muscles,
            indices=self.indices,
            index=11,
        )

    cpdef void log_active(
        self,
        unsigned int iteration,
        DTYPEv1 torques,
    ):
        """Log active torques"""
        log_torque(
            iteration=iteration,
            joints=self.joints,
            indices=self.indices,
            torques=torques,
            n_muscles=self.n_muscles,
            log_index=9,
        )

    cpdef void log_passive_stiffness(
        self,
        unsigned int iteration,
        DTYPEv1 torques,
    ):
        """Log passive stiffness"""
        log_torque(
            iteration=iteration,
            joints=self.joints,
            indices=self.indices,
            torques=torques,
            n_muscles=self.n_muscles,
            log_index=10,
        )

    cpdef void log_damping(
        self,
        unsigned int iteration,
        DTYPEv1 torques,
    ):
        """Log damping"""
        log_torque(
            iteration=iteration,
            joints=self.joints,
            indices=self.indices,
            torques=torques,
            n_muscles=self.n_muscles,
            log_index=11,
        )
