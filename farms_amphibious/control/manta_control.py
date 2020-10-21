"""Manta control"""

import numpy as np
from farms_bullet.model.control import ModelController, ControlType


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

    def step(self, iteration, time, timestep):
        """Control step"""
        self.animat_data.iteration = iteration

    def positions(self, iteration, time, timestep):
        """Postions"""
        n_joints = len(self.joints[ControlType.POSITION])
        positions = 5e-2*np.sin(2*np.pi*0.25*time)*np.ones(n_joints)
        positions[:n_joints//2] *= -1
        return dict(zip(
            self.joints[ControlType.POSITION],
            positions,
        ))

    def velocities(self, iteration, time, timestep):
        """Velocities"""
        return {}
