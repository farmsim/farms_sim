"""Position phase model"""

include 'sensor_convention.pxd'
cimport numpy as np
import numpy as np


cdef class PositionPhaseCy(JointsControlCy):
    """Position phase model"""

    def __init__(
            self,
            NetworkCy network,
            UITYPEv2 osc_indices,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.network = network
        self.osc_indices = osc_indices

    cpdef void step(self, unsigned int iteration):
        """Step"""
        cdef unsigned int joint_i, joint_data_i, osc_i
        cdef DTYPEv1 offsets = self.network.offsets(iteration)
        cdef DTYPEv1 phases = self.network.phases(iteration)

        # For each joint
        for joint_i in range(self.n_joints):

            # Data
            joint_data_i = self.indices[joint_i]
            osc_i = self.osc_indices[0][joint_data_i]

            assert osc_i < len(phases)

            # Position outputs
            self.joints_data.array[iteration, joint_data_i, JOINT_CMD_POSITION] = (
                self.transform_gain[joint_data_i]*(
                    phases[osc_i] + offsets[joint_data_i]
                ) + self.transform_bias[joint_data_i]
            )
