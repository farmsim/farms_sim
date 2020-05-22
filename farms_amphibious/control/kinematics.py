"""Kinematics"""

import numpy as np
from scipy.interpolate import interp1d
from farms_bullet.model.control import ModelController


def kinematics_interpolation(
        kinematics,
        sampling,
        timestep,
        n_iterations,
):
    """Kinematics interpolations"""
    data_duration = sampling*kinematics.shape[0]
    simulation_duration = timestep*n_iterations
    interp_x = np.arange(0, data_duration, sampling)
    interp_xn = np.arange(0, simulation_duration, timestep)
    assert data_duration >= simulation_duration, 'Data {} < {} Sim'.format(
        data_duration,
        simulation_duration
    )
    assert len(interp_x) == kinematics.shape[0]
    assert interp_x[-1] >= interp_xn[-1], 'Data[-1] {} < {} Sim[-1]'.format(
        interp_x[-1],
        interp_xn[-1]
    )
    return interp1d(
        interp_x,
        kinematics,
        axis=0
    )(interp_xn)


class AmphibiousKinematics(ModelController):
    """Amphibious kinematics"""

    def __init__(
            self,
            joints,
            animat_options,
            animat_data,
            timestep,
            n_iterations,
            sampling
    ):
        super(AmphibiousKinematics, self).__init__(
            joints=joints,
            use_position=True,
            use_torque=False,
        )
        kinematics = np.loadtxt(animat_options.control.kinematics_file)
        kinematics[:, 3:] = ((kinematics[:, 3:] + np.pi) % (2*np.pi)) - np.pi
        self.kinematics = kinematics_interpolation(
            kinematics=kinematics[:, 3:],
            sampling=sampling,
            timestep=timestep,
            n_iterations=n_iterations,
        )
        self.animat_options = animat_options
        self.animat_data = animat_data
        self._timestep = timestep
        max_torques = {
            joint.joint: joint.max_torque
            for joint in animat_options.control.joints
        }
        self.max_torques = np.array([
            max_torques[joint]
            for joint in joints
        ])

    def step(self, iteration, time, timestep):
        """Control step"""
        self.animat_data.iteration = iteration

    def get_outputs(self):
        """Outputs"""
        return self.kinematics[self.animat_data.iteration]

    def get_outputs_all(self):
        """Outputs"""
        return self.kinematics[:]

    def get_doutputs(self):
        """Outputs velocity"""
        return (
            (
                self.kinematics[self.animat_data.iteration]
                - self.kinematics[self.animat_data.iteration-1]
            )/self._timestep
            if self.animat_data.iteration
            else np.zeros_like(self.kinematics[0])
        )

    def get_doutputs_all(self):
        """Outputs velocity"""
        return np.diff(self.kinematics)

    def get_position_output(self):
        """Position output"""
        return self.get_outputs()

    def get_position_output_all(self):
        """Position output"""
        return self.get_outputs_all()

    def get_velocity_output(self):
        """Position output"""
        return self.get_doutputs()

    def get_velocity_output_all(self):
        """Position output"""
        return self.get_doutputs_all()

    def update(self, options):
        """Update drives"""

    def positions(self, iteration):
        """Postions"""
        return self.get_position_output()

    def velocities(self, iteration):
        """Postions"""
        return self.get_velocity_output()
