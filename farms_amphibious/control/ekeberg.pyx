"""Ekeberg muscle model"""

include 'types.pxd'
include 'sensor_convention.pxd'
cimport numpy as np
import numpy as np


cdef enum:

    ALPHA = 0
    BETA = 1
    GAMMA = 2
    DELTA = 3
    EPSILON = 4


cdef inline double sign(double value):
    """Sign"""
    if value < 0:
        return -1
    else:
        return 1


cdef class EkebergMuscleCy(JointsMusclesCy):
    """Ekeberg muscle model"""

    cpdef void step(self, unsigned int iteration):
        """Step"""
        cdef unsigned int joint_i, joint_data_i, osc_0, osc_1
        cdef DTYPE neural_diff, neural_sum
        cdef DTYPE active_torque, stiffness_intermediate
        cdef DTYPE active_stiffness, passive_stiffness, damping, friction
        cdef np.ndarray neural_activity = self.network.outputs(iteration)
        cdef np.ndarray joints_offsets = np.zeros(self.n_joints, dtype=np.double)
        cdef DTYPEv1 offsets = self.network.offsets(iteration)
        cdef DTYPEv1 positions = self.joints_data.positions(iteration)
        cdef DTYPEv1 velocities = self.joints_data.velocities(iteration)

        # For each muscle
        for joint_i in range(self.n_joints):

            joint_data_i = self.indices[joint_i]

            # Offsets
            joints_offsets[joint_i] = (
                self.transform_gain[joint_data_i]
                *offsets[joint_data_i]
                + self.transform_bias[joint_data_i]
            )

            # Data
            osc_0 = self.osc_indices[0][joint_i]
            osc_1 = self.osc_indices[1][joint_i]
            neural_diff = neural_activity[osc_0] - neural_activity[osc_1]
            neural_sum = neural_activity[osc_0] + neural_activity[osc_1]
            m_delta_phi = joints_offsets[joint_i] - positions[joint_data_i]

            # Torques
            active_torque = self.parameters[joint_i][ALPHA]*neural_diff
            stiffness_intermediate = (
                self.parameters[joint_i][BETA]
                *m_delta_phi
                *self.transform_gain[joint_data_i]
            )
            active_stiffness = neural_sum*stiffness_intermediate
            passive_stiffness = (
                self.parameters[joint_i][GAMMA]
                *stiffness_intermediate
            )
            damping = -(
                self.parameters[joint_i][DELTA]
                *velocities[joint_data_i]
                *self.transform_gain[joint_data_i]
            )
            friction = -(
                self.parameters[joint_i][EPSILON]
                *sign(velocities[joint_data_i])
            )

            # Log
            torque = active_torque + active_stiffness + passive_stiffness + damping + friction
            self.joints_data.array[iteration, joint_data_i, JOINT_CMD_TORQUE] = torque
            self.joints_data.array[iteration, joint_data_i, JOINT_TORQUE_ACTIVE] = active_torque + active_stiffness
            self.joints_data.array[iteration, joint_data_i, JOINT_TORQUE_STIFFNESS] = passive_stiffness
            self.joints_data.array[iteration, joint_data_i, JOINT_TORQUE_DAMPING] = damping
            self.joints_data.array[iteration, joint_data_i, JOINT_TORQUE_FRICTION] = friction
