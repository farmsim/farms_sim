"""Network"""

import numpy as np
from scipy import integrate
from farms_bullet.model.control import ModelController
from ..controllers.controller import ode_oscillators_sparse
from ..model.convention import AmphibiousConvention


class NetworkODE:
    """NetworkODE"""

    def __init__(self, data, n_oscillators):
        super(NetworkODE, self).__init__()
        self.ode = ode_oscillators_sparse
        self.data = data
        self.n_oscillators = n_oscillators

        # Adaptive timestep parameters
        self.solver = integrate.ode(f=self.ode)  # , jac=self.jac
        self.solver.set_integrator("dopri5")
        self.solver.set_f_params(self.data)
        self._time = 0

    def control_step(self, iteration, time, timestep):
        """Control step"""
        self.data.iteration = iteration
        self.solver.set_initial_value(
            self.data.state.array[iteration, 0, :],
            time
        )
        self.data.state.array[iteration+1, 0, :] = (
            self.solver.integrate(time+timestep)
        )

    def phases(self):
        """Oscillators phases"""
        return self.data.state.array[:, 0, :self.n_oscillators]

    def dphases(self):
        """Oscillators phases velocity"""
        return self.data.state.array[:, 1, :self.n_oscillators]

    def amplitudes(self):
        """Amplitudes"""
        return self.data.state.array[:, 0, self.n_oscillators:2*self.n_oscillators]

    def damplitudes(self):
        """Amplitudes velocity"""
        return self.data.state.array[:, 1, self.n_oscillators:2*self.n_oscillators]

    def get_outputs(self, iteration):
        """Outputs"""
        return self.amplitudes()[iteration]*(
            1 + np.cos(self.phases()[iteration])
        )

    def get_outputs_all(self):
        """Outputs"""
        return self.amplitudes()*(
            1 + np.cos(self.phases())
        )

    def get_doutputs(self, iteration):
        """Outputs velocity"""
        return self.damplitudes()[iteration]*(
            1 + np.cos(self.phases()[iteration])
        ) - (
            self.amplitudes()[iteration]
            *np.sin(self.phases()[iteration])
            *self.dphases()[iteration]
        )

    def get_doutputs_all(self):
        """Outputs velocity"""
        return self.damplitudes()*(
            1 + np.cos(self.phases)
        ) - self.amplitudes*np.sin(self.phases)*self.dphases

    def offsets(self):
        """Offset"""
        return self.data.state.array[:, 0, 2*self.n_oscillators:]

    def doffsets(self):
        """Offset velocity"""
        return self.data.state.array[:, 1, 2*self.n_oscillators:]


class AmphibiousController(ModelController):
    """Amphibious network"""

    def __init__(self, joints, animat_options, animat_data, timestep):
        convention = AmphibiousConvention(animat_options.morphology)
        super(AmphibiousController, self).__init__(
            joints=joints,
            use_position=True,
            use_torque=False,
        )
        self.network = NetworkODE(
            data=animat_data,
            n_oscillators=animat_data.state.n_oscillators,
        )
        self.animat_data = animat_data
        self._timestep = timestep
        n_body = animat_options.morphology.n_joints_body
        n_legs_dofs = animat_options.morphology.n_dof_legs
        self.groups = [None, None]
        self.groups = [
            [
                convention.bodyosc2index(
                    joint_i=i,
                    side=side
                )
                for i in range(n_body)
            ] + [
                convention.legosc2index(
                    leg_i=leg_i,
                    side_i=side_i,
                    joint_i=joint_i,
                    side=side
                )
                for leg_i in range(animat_options.morphology.n_legs//2)
                for side_i in range(2)
                for joint_i in range(n_legs_dofs)
            ]
            for side in range(2)
        ]
        self.gain_amplitude = np.array(
            animat_options.control.network.joints.gain_amplitude
        )
        self.gain_offset = np.array(
            animat_options.control.network.joints.gain_offset
        )

    def control_step(self, iteration, time, timestep):
        """Control step"""
        self.network.control_step(iteration, time, timestep)

    def get_position_output(self, iteration):
        """Position output"""
        outputs = self.network.get_outputs(iteration)
        return (
            self.gain_amplitude*0.5*(
                outputs[self.groups[0]]
                - outputs[self.groups[1]]
            )
            + self.gain_offset*self.network.offsets()[iteration]
        )

    def get_position_output_all(self):
        """Position output"""
        outputs = self.network.get_outputs_all()
        return (
            self.gain_amplitude*0.5*(
                outputs[:, self.groups[0]]
                - outputs[:, self.groups[1]]
            )
            + self.gain_offset*self.network.offsets()
        )

    def get_velocity_output(self, iteration):
        """Position output"""
        outputs = self.network.get_doutputs(iteration)
        return (
            self.gain_amplitude*0.5*(
                outputs[self.groups[0]]
                - outputs[self.groups[1]]
            )
            + self.network.doffsets()[iteration]
        )

    def get_velocity_output_all(self):
        """Position output"""
        outputs = self.network.get_doutputs_all()
        return self.gain_amplitude*0.5*(
            outputs[:, self.groups[0]]
            - outputs[:, self.groups[1]]
        )

    def get_torque_output(self, iteration):
        """Torque output"""
        proprioception = self.animat_data.sensors.proprioception
        positions = np.array(proprioception.positions(iteration))
        velocities = np.array(proprioception.velocities(iteration))
        predicted_positions = (positions+3*self._timestep*velocities)
        cmd_positions = self.get_position_output(iteration)
        cmd_velocities = self.get_velocity_output(iteration)
        positions_rest = np.array(self.network.offsets()[iteration])
        cmd_kp = 1e1  # Nm/rad
        cmd_kd = 1e-2  # Nm*s/rad
        spring = 1e0  # Nm/rad
        damping = 1e-2  # Nm*s/rad
        max_torque = 1  # Nm
        torques = np.clip(
            (
                + cmd_kp*(cmd_positions-predicted_positions)
                + cmd_kd*(cmd_velocities-velocities)
                + spring*(positions_rest-predicted_positions)
                - damping*velocities
            ),
            -max_torque,
            +max_torque
        )
        return torques

    def positions(self, iteration):
        """Postions"""
        return self.get_position_output(iteration)

    def velocities(self, iteration):
        """Postions"""
        return self.get_velocity_output(iteration)

    def update(self, options):
        """Update drives"""
        self.animat_data.network.oscillators.update(options)
        self.animat_data.joints.update(options)
