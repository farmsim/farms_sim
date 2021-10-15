"""Joints muscles"""

include 'sensor_convention.pxd'
cimport numpy as np
import numpy as np
from .joints_control_cy cimport get_joints_data, set_joints_data


cdef class JointsMusclesCy(JointsControlCy):
    """Joints muscles"""

    def __init__(
            self,
            NetworkCy network,
            DTYPEv2 parameters,
            UITYPEv2 osc_indices,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.network = network
        self.parameters = parameters
        self.osc_indices = osc_indices

    cpdef np.ndarray torques_implicit(self, unsigned int iteration):
        """Torques"""
        cdef unsigned int muscle_i, joint_i
        cdef np.ndarray torques = np.zeros(self.n_joints, dtype=np.double)
        for muscle_i in range(self.n_joints):
            joint_i = self.indices[muscle_i]
            torques[muscle_i] = (
                self.joints_data.array[iteration, joint_i, JOINT_TORQUE_ACTIVE]
                + self.joints_data.array[iteration, joint_i, JOINT_TORQUE_STIFFNESS]
            )
        return torques

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

    cpdef void set_active(
        self,
        unsigned int iteration,
        DTYPEv1 data,
    ):
        """Set active torques"""
        set_joints_data(
            iteration=iteration,
            joints_data=self.joints_data,
            n_joints=self.n_joints,
            indices=self.indices,
            array_index=JOINT_TORQUE_ACTIVE,
            data=data,
        )

    cpdef void set_passive_stiffness(
        self,
        unsigned int iteration,
        DTYPEv1 data,
    ):
        """Set passive stiffness"""
        set_joints_data(
            iteration=iteration,
            joints_data=self.joints_data,
            n_joints=self.n_joints,
            indices=self.indices,
            array_index=JOINT_TORQUE_STIFFNESS,
            data=data,
        )

    cpdef void set_damping(
        self,
        unsigned int iteration,
        DTYPEv1 data,
    ):
        """Set damping"""
        set_joints_data(
            iteration=iteration,
            joints_data=self.joints_data,
            indices=self.indices,
            n_joints=self.n_joints,
            array_index=JOINT_TORQUE_DAMPING,
            data=data,
        )

    cpdef void set_friction(
        self,
        unsigned int iteration,
        DTYPEv1 data,
    ):
        """Set friction"""
        set_joints_data(
            iteration=iteration,
            joints_data=self.joints_data,
            indices=self.indices,
            n_joints=self.n_joints,
            array_index=JOINT_TORQUE_FRICTION,
            data=data,
        )
