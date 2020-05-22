"""Network controller"""

import numpy as np
from farms_bullet.model.control import ModelController, ControlType
from ..model.convention import AmphibiousConvention
from .network import NetworkODE


class AmphibiousController(ModelController):
    """Amphibious network"""

    def __init__(self, joints, animat_options, animat_data, timestep):
        convention = AmphibiousConvention(**animat_options.morphology)
        super(AmphibiousController, self).__init__(
            joints=joints,
            control_types={
                joint.joint: joint.control_type
                for joint in animat_options.control.joints
            },
            max_torques={
                joint.joint: joint.max_torque
                for joint in animat_options.control.joints
            }
        )
        self.network = NetworkODE(animat_data)
        self.animat_data = animat_data
        self._timestep = timestep
        n_body = animat_options.morphology.n_joints_body
        n_legs_dofs = animat_options.morphology.n_dof_legs
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
        gain_amplitudes = {
            joint.joint: joint.gain_amplitude
            for joint in animat_options.control.joints
        }
        self.gain_amplitude = np.array([
            gain_amplitudes[joint]
            for joint in joints
        ])
        gain_offsets = {
            joint.joint: joint.gain_offset
            for joint in animat_options.control.joints
        }
        self.gain_offset = np.array([
            gain_offsets[joint]
            for joint in joints
        ])
        offsets_bias = {
            joint.joint: joint.bias
            for joint in animat_options.control.joints
        }
        self.joints_bias = np.array([
            offsets_bias[joint]
            for joint in joints
        ])

    def step(self, iteration, time, timestep):
        """Control step"""
        self.network.step(iteration, time, timestep)

    def positions(self, iteration):
        """Postions"""
        outputs = self.network.outputs(iteration)
        positions = (
            self.gain_amplitude*0.5*(
                outputs[self.groups[0]]
                - outputs[self.groups[1]]
            )
            + self.gain_offset*self.network.offsets(iteration)
            + self.joints_bias
        )
        return dict(zip(self.joints[ControlType.POSITION], positions))

    def torques(self, iteration):
        """Torques"""
        proprioception = self.animat_data.sensors.proprioception
        positions = np.array(proprioception.positions(iteration))
        velocities = np.array(proprioception.velocities(iteration))
        outputs = self.network.outputs(iteration)
        cmd_positions = (
            self.gain_amplitude*0.5*(
                outputs[self.groups[0]]
                - outputs[self.groups[1]]
            )
            + self.gain_offset*self.network.offsets(iteration)
            + self.joints_bias
        )
        # cmd_velocities = self.get_velocity_output(iteration)
        positions_rest = np.array(self.network.offsets()[iteration])
        # max_torque = 1  # Nm
        spring = 2e0  # Nm/rad
        damping = 5e-3  # max_torque/10  # 1e-1 # Nm*s/rad
        cmd_kp = 5*spring  # Nm/rad
        # cmd_kd = 0.5*damping  # Nm*s/rad
        motor_torques = cmd_kp*(cmd_positions-positions)
        spring_torques = spring*(positions_rest-positions)
        damping_torques = - damping*velocities
        # if iteration > 0:
        #     motor_torques += cmd_kd*(
        #         (cmd_positions - self.positions(iteration-1))/self._timestep
        #         - velocities
        #     )
        torques = motor_torques + spring_torques + damping_torques
        proprioception.array[iteration, :, 8] = torques
        proprioception.array[iteration, :, 9] = motor_torques
        proprioception.array[iteration, :, 10] = spring_torques
        proprioception.array[iteration, :, 11] = damping_torques
        return dict(zip(self.joints[ControlType.TORQUE], torques))
