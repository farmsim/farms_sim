"""Position muscle model"""

include 'types.pxd'
include 'sensor_convention.pxd'
cimport numpy as np
import numpy as np


cdef class PositionMuscleCy(JointsMusclesCy):
    """Position muscle model"""

    cpdef void step(self, unsigned int iteration):
        """Step"""
        cdef unsigned int joint_i, joint_data_i, osc_0, osc_1
        cdef DTYPE neural_diff
        cdef np.ndarray neural_activity = self.network.outputs(iteration)
        cdef DTYPEv1 offsets = self.network.offsets(iteration)

        # For each joint
        for joint_i in range(self.n_joints):

            # Data
            joint_data_i = self.indices[joint_i]
            osc_0 = self.osc_indices[0][joint_data_i]
            osc_1 = self.osc_indices[1][joint_data_i]
            neural_diff = neural_activity[osc_0] - neural_activity[osc_1]

            # Position outputs
            self.joints_data.array[iteration, joint_data_i, JOINT_CMD_POSITION] = (
                self.transform_gain[joint_data_i]
                *(0.5*neural_diff + offsets[joint_data_i])
                + self.transform_bias[joint_data_i]
            )
