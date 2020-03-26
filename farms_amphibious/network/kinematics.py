"""Kinematics"""

import numpy as np
from scipy.interpolate import interp1d
from farms_bullet.model.control import ModelController


class AmphibiousKinematics(ModelController):
    """Amphibious kinematics"""

    def __init__(self, animat_options, animat_data, timestep, n_iterations, sampling):
        super(AmphibiousKinematics, self).__init__(
            joints=np.zeros(animat_options.morphology.n_joints()),
            use_position=True,
            use_torque=False,
        )
        self.kinematics = np.loadtxt(animat_options.control.kinematics_file)
        self.kinematics = self.kinematics[:, 3:]
        self.kinematics = ((self.kinematics + np.pi) % (2*np.pi)) - np.pi
        data_duration = sampling*self.kinematics.shape[0]
        simulation_duration = timestep*n_iterations
        interp_x = np.arange(0, data_duration, sampling)
        interp_xn = np.arange(0, simulation_duration, timestep)
        assert data_duration >= simulation_duration, 'Data {} < {} Sim'.format(
            data_duration,
            simulation_duration
        )
        assert len(interp_x) == self.kinematics.shape[0]
        assert interp_x[-1] >= interp_xn[-1], 'Data[-1] {} < {} Sim[-1]'.format(
            interp_x[-1],
            interp_xn[-1]
        )
        self.kinematics = interp1d(
            interp_x,
            self.kinematics,
            axis=0
        )(interp_xn)
        self.animat_options = animat_options
        self.animat_data = animat_data
        self._timestep = timestep

    def control_step(self):
        """Control step"""
        self.animat_data.iteration += 1

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

    def positions(self):
        """Postions"""
        return self.get_position_output()

    def velocities(self):
        """Postions"""
        return self.get_velocity_output()
