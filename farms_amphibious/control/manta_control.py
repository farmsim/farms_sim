"""Manta control"""

import numpy as np
from farms_bullet.model.control import ModelController, ControlType

def joints_sorted(names):
    """Joints sorted"""
    left = sorted([
        joint
        for joint in names
        if 'left' in joint
    ])
    right = sorted([
        joint
        for joint in names
        if 'right' in joint
    ])
    passive_left = sorted([
        joint
        for joint in names
        if 'passive' in joint and 'left' in joint
    ])
    passive_right = sorted([
        joint
        for joint in names
        if 'passive' in joint and 'right' in joint
    ])
    return left, right, passive_left, passive_right



def control(time, left, right, passive_left, passive_right):
    """Control"""
    value = 5e-2*np.sin(2*np.pi*0.25*time+0.5*np.pi)
    value2 = 1e-1*np.sin(2*np.pi*0.25*time)
    n_left = len(passive_left)
    n_right = len(passive_right)
    return dict([
        [joint, -value]
        for joint in left
    ] + [
        [joint, value]
        for joint in right
    ] + [
        [joint, joint_i*value2/n_left]
        for joint_i, joint in enumerate(passive_left)
    ] + [
        [joint, joint_i*value2/n_right]
        for joint_i, joint in enumerate(passive_right)
    ])


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
