"""Network controller"""

import numpy as np
from farms_bullet.model.control import ModelController, ControlType
from .network import NetworkODE


class AmphibiousController(ModelController):
    """Amphibious network"""

    def __init__(self, joints, animat_options, animat_data):
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
        control_types = [
            ControlType.POSITION,
            ControlType.VELOCITY,
            ControlType.TORQUE,
        ]
        joint_muscle_map = {
            muscle.joint: muscle
            for muscle in animat_options.control.muscles
        }
        muscles = [
            [joint_muscle_map[joint] for joint in self.joints[control_type]]
            for control_type in control_types
        ]
        self.alphas = [
            np.array([muscle.alpha for muscle in muscles[control_type]])
            for control_type in control_types
        ]
        self.betas = [
            np.array([muscle.beta for muscle in muscles[control_type]])
            for control_type in control_types
        ]
        self.gammas = [
            np.array([muscle.gamma for muscle in muscles[control_type]])
            for control_type in control_types
        ]
        self.deltas = [
            np.array([muscle.delta for muscle in muscles[control_type]])
            for control_type in control_types
        ]
        osc_map = {}
        for muscle in animat_options.control.muscles:
            osc_map[muscle.osc1] = (
                self.animat_data.network.oscillators.names.index(muscle.osc1)
            )
            osc_map[muscle.osc2] = (
                self.animat_data.network.oscillators.names.index(muscle.osc2)
            )
        self.muscle_groups = [
            [
                [osc_map[muscle.osc1] for muscle in muscles[control_type]],
                [osc_map[muscle.osc2] for muscle in muscles[control_type]],
            ]
            for control_type in control_types
        ]
        gain_amplitudes = {
            joint.joint: joint.gain_amplitude
            for joint in animat_options.control.joints
        }
        self.gain_amplitude = np.array([
            gain_amplitudes[joint]
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
            self.gain_amplitude*(
                0.5*(
                    outputs[self.muscle_groups[ControlType.POSITION][0]]
                    - outputs[self.muscle_groups[ControlType.POSITION][1]]
                )
                + self.network.offsets(iteration)
            ) + self.joints_bias
        )
        return dict(zip(self.joints[ControlType.POSITION], positions))

    def pid_controller(self, iteration):
        """Torques"""
        proprioception = self.animat_data.sensors.proprioception
        positions = np.array(proprioception.positions(iteration))
        velocities = np.array(proprioception.velocities(iteration))
        outputs = self.network.outputs(iteration)
        cmd_positions = (
            self.gain_amplitude*0.5*(
                outputs[self.muscle_groups[0]]
                - outputs[self.muscle_groups[1]]
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
        torques = np.clip(
            motor_torques + spring_torques + damping_torques,
            -self.max_torques[ControlType.TORQUE],
            self.max_torques[ControlType.TORQUE],
        )
        proprioception.array[iteration, :, 8] = torques
        proprioception.array[iteration, :, 9] = motor_torques
        proprioception.array[iteration, :, 10] = spring_torques
        proprioception.array[iteration, :, 11] = damping_torques
        return dict(zip(self.joints[ControlType.TORQUE], torques))

    def ekeberg_muscle(self, iteration):
        """Ekeberg muscle"""
        # Sensors
        proprioception = self.animat_data.sensors.proprioception
        positions = np.array(proprioception.positions(iteration))
        velocities = np.array(proprioception.velocities(iteration))

        # Neural activity
        neural_activity = self.network.outputs(iteration)

        # Joints offsets
        joints_offsets = (
            self.gain_amplitude*self.network.offsets(iteration)
            + self.joints_bias
        )

        # Torques
        neural_diff = (
            neural_activity[self.muscle_groups[ControlType.TORQUE][0]]
            - neural_activity[self.muscle_groups[ControlType.TORQUE][1]]
        )
        neural_sum = (
            neural_activity[self.muscle_groups[ControlType.TORQUE][0]]
            + neural_activity[self.muscle_groups[ControlType.TORQUE][1]]
        )
        active_torques = (
            self.gain_amplitude*self.alphas[ControlType.TORQUE]*neural_diff
        )
        stiffness_torques = self.betas[ControlType.TORQUE]*(
            neural_sum + self.gammas[ControlType.TORQUE]
        )*(positions - joints_offsets)
        damping_torques = self.deltas[ControlType.TORQUE]*velocities

        # Final torques
        torques = np.clip(
            active_torques + stiffness_torques + damping_torques,
            -self.max_torques[ControlType.TORQUE],
            self.max_torques[ControlType.TORQUE],
        )
        proprioception.array[iteration, :, 8] = torques
        proprioception.array[iteration, :, 9] = active_torques
        proprioception.array[iteration, :, 10] = stiffness_torques
        proprioception.array[iteration, :, 11] = damping_torques
        return dict(zip(self.joints[ControlType.TORQUE], torques))

    def torques(self, iteration):
        """Torques"""
        return self.ekeberg_muscle(iteration)
