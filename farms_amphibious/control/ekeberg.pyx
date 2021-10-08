"""Ekeberg muscle model"""

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


cdef class EkebergMuscleCy:
    """SensorsData"""

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

    cpdef void step(self, unsigned int iteration):
        """Step"""
        cdef unsigned int muscle_i, joint_i, osc_0, osc_1
        cdef DTYPE neural_diff, neural_sum
        cdef DTYPE active_torque, stiffness_intermediate
        cdef DTYPE active_stiffness, passive_stiffness, damping
        cdef np.ndarray neural_activity = self.network.outputs(iteration)
        cdef np.ndarray joints_offsets = np.zeros(self.n_muscles, dtype=np.double)
        cdef DTYPEv1 offsets = self.network.offsets(iteration)
        cdef DTYPEv1 positions = self.joints.positions(iteration)
        cdef DTYPEv1 velocities = self.joints.velocities(iteration)

        # For each muscle
        for muscle_i in range(self.n_muscles):

            joint_i = self.indices[muscle_i]

            # Offsets
            joints_offsets[muscle_i] = (
                self.transform_gain[joint_i]
                *offsets[joint_i]
                + self.transform_bias[joint_i]
            )

            # Data
            osc_0 = self.groups[0][muscle_i]
            osc_1 = self.groups[1][muscle_i]
            neural_diff = neural_activity[osc_0] - neural_activity[osc_1]
            neural_sum = neural_activity[osc_0] + neural_activity[osc_1]
            m_delta_phi = joints_offsets[muscle_i] - positions[joint_i]

            # Torques
            active_torque = self.parameters[muscle_i][0]*neural_diff
            stiffness_intermediate = self.parameters[muscle_i][1]*m_delta_phi
            active_stiffness = neural_sum*stiffness_intermediate
            passive_stiffness = self.parameters[muscle_i][2]*stiffness_intermediate
            damping = -self.parameters[muscle_i][3]*velocities[joint_i]

            # Log
            torque = active_torque + active_stiffness + passive_stiffness + damping
            self.joints.array[iteration, joint_i, 8] = torque
            self.joints.array[iteration, joint_i, 9] = active_torque + active_stiffness
            self.joints.array[iteration, joint_i, 10] = passive_stiffness
            self.joints.array[iteration, joint_i, 11] = damping

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
