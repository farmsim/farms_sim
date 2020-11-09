"""Manta control"""

import numpy as np
from farms_bullet.model.control import ModelController, ControlType


def joints_sorted(names):
    """Joints sorted"""
    left = sorted([
        joint
        for joint in names
        if 'left' in joint
        and 'passive' not in joint
    ])
    right = sorted([
        joint
        for joint in names
        if 'right' in joint
        and 'passive' not in joint
    ])
    n_joints = len(left)
    passive_left = [
        sorted([
            joint
            for joint in names
            if 'joint_{:02d}_passive_left'.format(joint_i) in joint
        ])
        for joint_i in range(n_joints)
    ]
    passive_right = [
        sorted([
            joint
            for joint in names
            if 'joint_{:02d}_passive_right'.format(n_joints+joint_i) in joint
        ])
        for joint_i in range(n_joints)
    ]
    joints = (
        left + right
        + [j for _j in passive_left for j in _j]
        + [j for _j in passive_right for j in _j]
    )
    for name in names:
        assert name in joints
    return left, right, passive_left, passive_right


def control(time, left, right, passive_left, passive_right):
    """Control"""
    value = 5e-2*np.sin(2*np.pi*0.5*time+0.5*np.pi)
    value2 = 3e-1*np.sin(2*np.pi*0.5*time)
    n_left = len(left)
    n_right = len(right)
    n_left_long = len(passive_left[0])
    n_right_long = len(passive_right[0])
    joints = dict([
        [joint, -value]
        for joint in left
    ] + [
        [joint, value]
        for joint in right
    ] + [
        [joint, joint_i*joint_j*value2/(n_left*n_left_long)]
        for joint_i, _joints in enumerate(passive_left)
        for joint_j, joint in enumerate(_joints)
    ] + [
        [joint, joint_i*joint_j*value2/(n_right*n_right_long)]
        for joint_i, _joints in enumerate(passive_right)
        for joint_j, joint in enumerate(_joints)
    ])
    return joints


class MantaController(ModelController):
    """Manta controller"""

    def __init__(
            self,
            joints,
            animat_options,
            animat_data,
    ):
        super(MantaController, self).__init__(
            joints=joints,
            control_types={joint: ControlType.POSITION for joint in joints},
            max_torques={joint: 1e3 for joint in joints},
        )
        self.animat_options = animat_options
        self.animat_data = animat_data
        (
            self.joints_left,
            self.joints_right,
            self.joints_passive_left,
            self.joints_passive_right,
        ) = joints_sorted(names=self.joints[ControlType.POSITION])

    def step(self, iteration, time, timestep):
        """Control step"""
        self.animat_data.iteration = iteration

    def positions(self, iteration, time, timestep):
        """Postions"""
        return control(
            time,
            self.joints_left,
            self.joints_right,
            self.joints_passive_left,
            self.joints_passive_right,
        )

    def velocities(self, iteration, time, timestep):
        """Velocities"""
        return {}
