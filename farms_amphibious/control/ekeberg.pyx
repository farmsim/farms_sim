"""Ekeberg muscle model"""

cimport numpy as np
import numpy as np

from .network_cy cimport NetworkCy


cdef class EkebergMuscleCy(MuscleCy):
    """Ekeberg muscle model"""

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
