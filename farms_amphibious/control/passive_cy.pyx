"""Passive muscle model"""

include 'sensor_convention.pxd'
cimport numpy as np
import numpy as np
from .joints_control_cy cimport get_joints_data, set_joints_data


cdef inline double sign(double value):
    """Sign"""
    if value < 0:
        return -1
    else:
        return 1


cdef class PassiveJointCy(JointsControlCy):
    """Passive muscle model"""

    def __init__(
            self,
            stiffness_coefficients,
            damping_coefficients,
            friction_coefficients,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.stiffness_coefficients = stiffness_coefficients
        self.damping_coefficients = damping_coefficients
        self.friction_coefficients = friction_coefficients

    cpdef void step(self, unsigned int iteration):
        """Step"""
        cdef unsigned int joint_data_i
        cdef DTYPE stiffness, damping, friction
        cdef DTYPEv1 positions = self.joints_data.positions(iteration)
        cdef DTYPEv1 velocities = self.joints_data.velocities(iteration)

        # For each muscle
        for joint_i in range(self.n_joints):

            # Joint index
            joint_data_i = self.indices[joint_i]

            # Torques
            passive_stiffness = -self.stiffness_coefficients[joint_i]*(
                positions[joint_data_i] - self.transform_bias[joint_data_i]
            )*self.transform_gain[joint_data_i]
            damping = -(
                self.damping_coefficients[joint_i]
                *velocities[joint_data_i]
                *self.transform_gain[joint_data_i]
            )
            friction = -(
                self.friction_coefficients[joint_i]
                *sign(velocities[joint_data_i])
                *self.transform_gain[joint_data_i]
            )

            # Log
            self.joints_data.array[iteration, joint_data_i, JOINT_TORQUE] = passive_stiffness + damping + friction
            self.joints_data.array[iteration, joint_data_i, JOINT_TORQUE_STIFFNESS] = passive_stiffness
            self.joints_data.array[iteration, joint_data_i, JOINT_TORQUE_DAMPING] = damping
            self.joints_data.array[iteration, joint_data_i, JOINT_TORQUE_FRICTION] = friction

    cpdef np.ndarray stiffness(self, unsigned int iteration):
        """Torques"""
        return get_joints_data(
            iteration=iteration,
            joints_data=self.joints_data,
            n_joints=self.n_joints,
            indices=self.indices,
            array_index=JOINT_TORQUE_STIFFNESS,
        )

    cpdef np.ndarray damping(self, unsigned int iteration):
        """Torques"""
        return get_joints_data(
            iteration=iteration,
            joints_data=self.joints_data,
            n_joints=self.n_joints,
            indices=self.indices,
            array_index=JOINT_TORQUE_DAMPING,
        )

    cpdef np.ndarray friction(self, unsigned int iteration):
        """Torques"""
        return get_joints_data(
            iteration=iteration,
            joints_data=self.joints_data,
            n_joints=self.n_joints,
            indices=self.indices,
            array_index=JOINT_TORQUE_FRICTION,
        )
