"""Ekeberg muscle model"""

include 'types.pxd'
include 'sensor_convention.pxd'
cimport numpy as np
import numpy as np


cdef class EkebergMuscleCy(JointsMusclesCy):
    """Ekeberg muscle model"""

    cpdef void step(self, unsigned int iteration):
        """Step"""
        cdef unsigned int muscle_i, joint_i, osc_0, osc_1
        cdef DTYPE neural_diff, neural_sum
        cdef DTYPE active_torque, stiffness_intermediate
        cdef DTYPE active_stiffness, passive_stiffness, damping
        cdef np.ndarray neural_activity = self.network.outputs(iteration)
        cdef np.ndarray joints_offsets = np.zeros(self.n_joints, dtype=np.double)
        cdef DTYPEv1 offsets = self.network.offsets(iteration)
        cdef DTYPEv1 positions = self.joints_data.positions(iteration)
        cdef DTYPEv1 velocities = self.joints_data.velocities(iteration)

        # For each muscle
        for muscle_i in range(self.n_joints):

            joint_i = self.indices[muscle_i]

            # Offsets
            joints_offsets[muscle_i] = (
                self.transform_gain[joint_i]
                *offsets[joint_i]
                + self.transform_bias[joint_i]
            )

            # Data
            osc_0 = self.osc_indices[0][muscle_i]
            osc_1 = self.osc_indices[1][muscle_i]
            neural_diff = neural_activity[osc_0] - neural_activity[osc_1]
            neural_sum = neural_activity[osc_0] + neural_activity[osc_1]
            m_delta_phi = joints_offsets[muscle_i] - positions[joint_i]

            # Torques
            active_torque = self.parameters[muscle_i][0]*neural_diff
            stiffness_intermediate = (
                self.parameters[muscle_i][1]
                *m_delta_phi
                *self.transform_gain[joint_i]
            )
            active_stiffness = neural_sum*stiffness_intermediate
            passive_stiffness = self.parameters[muscle_i][2]*stiffness_intermediate
            damping = -(
                self.parameters[muscle_i][3]
                *velocities[joint_i]
                *self.transform_gain[joint_i]
            )

            # Log
            torque = active_torque + active_stiffness + passive_stiffness + damping
            self.joints_data.array[iteration, joint_i, JOINT_CMD_TORQUE] = torque
            self.joints_data.array[iteration, joint_i, JOINT_TORQUE_ACTIVE] = active_torque + active_stiffness
            self.joints_data.array[iteration, joint_i, JOINT_TORQUE_STIFFNESS] = passive_stiffness
            self.joints_data.array[iteration, joint_i, JOINT_TORQUE_DAMPING] = damping
