"""Network controller"""

import numpy as np
from farms_bullet.model.control import ModelController, ControlType
from .network import NetworkODE


class AmphibiousController(ModelController):
    """Amphibious network"""

    def __init__(self, joints, animat_options, animat_data):
        super().__init__(
            joints=joints,
            control_types={
                joint.joint: joint.control_type
                for joint in animat_options.control.joints
            },
            max_torques={
                joint.joint: joint.max_torque
                for joint in animat_options.control.joints
            },
        )
        self.network = NetworkODE(animat_data)
        self.animat_data = animat_data
        self.joints_map = JointsMap(
            joints=self.joints,
            joints_names=joints,
            animat_options=animat_options,
            control_types=self.control_types,
        )

        # Torque equation
        self.torque_equation = {
            'passive': self.passive,
            'ekeberg_muscle': self.ekeberg_muscle,
        }[animat_options.control.torque_equation]

        # Muscles
        joint_muscle_map = {
            muscle.joint: muscle
            for muscle in animat_options.control.muscles
        }
        muscles = [
            [joint_muscle_map[joint] for joint in self.joints[control_type]]
            for control_type in self.control_types
        ]
        self.alphas = [
            np.array([muscle.alpha for muscle in muscles[control_type]])
            for control_type in self.control_types
        ]
        self.betas = [
            np.array([muscle.beta for muscle in muscles[control_type]])
            for control_type in self.control_types
        ]
        self.gammas = [
            np.array([muscle.gamma for muscle in muscles[control_type]])
            for control_type in self.control_types
        ]
        self.deltas = [
            np.array([muscle.delta for muscle in muscles[control_type]])
            for control_type in self.control_types
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
            for control_type in self.control_types
        ]

    def step(self, iteration, time, timestep):
        """Control step"""
        self.network.step(iteration, time, timestep)

    def positions(self, iteration, time, timestep):
        """Postions"""
        outputs = self.network.outputs(iteration)
        positions = (
            self.joints_map.gain_amplitude[ControlType.POSITION]*(
                0.5*(
                    outputs[self.muscle_groups[ControlType.POSITION][0]]
                    - outputs[self.muscle_groups[ControlType.POSITION][1]]
                )
                + np.array(self.network.offsets(iteration))[
                    self.joints_map.indices[ControlType.POSITION]
                ]
            ) + self.joints_map.bias[ControlType.POSITION]
        )
        return dict(zip(self.joints[ControlType.POSITION], positions))

    def ekeberg_muscle(self, iteration, time, timestep, use_prediction=False):
        """Ekeberg muscle"""

        # Sensors
        joints = self.animat_data.sensors.joints
        positions = np.array(joints.positions(iteration))[
            self.joints_map.indices[ControlType.TORQUE]
        ]
        velocities = np.array(joints.velocities(iteration))[
            self.joints_map.indices[ControlType.TORQUE]
        ]
        if use_prediction:
            n_iters = 1
            velocities_prev = (
                np.array(joints.velocities(iteration-n_iters))[
                    self.joints_map.indices[ControlType.TORQUE]
                ]
                if iteration > n_iters
                else velocities
            )
            acceleration = (velocities - velocities_prev)/(n_iters*timestep)
            positions = positions + 0.5*timestep*velocities  #  + (
            #     0.125*acceleration*timestep**2
            # )
            velocities = velocities + 0.5*acceleration*timestep

        # Neural activity
        neural_activity = self.network.outputs(iteration)

        # Joints offsets
        joints_offsets = (
            self.joints_map.gain_amplitude[ControlType.TORQUE]
            *np.array(self.network.offsets(iteration))[
                self.joints_map.indices[ControlType.TORQUE]
            ]
            + self.joints_map.bias[ControlType.TORQUE]
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
            self.joints_map.gain_amplitude[ControlType.TORQUE]
            *self.alphas[ControlType.TORQUE]
            *neural_diff
        )
        active_stiffness = self.betas[ControlType.TORQUE]*(
            neural_sum
        )*(positions - joints_offsets)
        passive_stiffness = self.betas[ControlType.TORQUE]*(
            self.gammas[ControlType.TORQUE]
        )*(positions - joints_offsets)
        damping = self.deltas[ControlType.TORQUE]*velocities

        # Final torques
        torques = np.clip(
            active_torques + active_stiffness + passive_stiffness + damping,
            -self.max_torques[ControlType.TORQUE],
            self.max_torques[ControlType.TORQUE],
        )
        for i, idx in enumerate(self.joints_map.indices[ControlType.TORQUE]):
            joints.array[iteration, idx, 8] = torques[i]
            joints.array[iteration, idx, 9] = active_torques[i] + active_stiffness[i]
            joints.array[iteration, idx, 10] = passive_stiffness[i]
            joints.array[iteration, idx, 11] = damping[i]
        return dict(zip(self.joints[ControlType.TORQUE], torques))

    def passive(self, iteration, time, timestep):
        """Passive joints"""
        joints = self.animat_data.sensors.joints
        positions = np.array(joints.positions(iteration))[
            self.joints_map.indices[ControlType.TORQUE]
        ]
        velocities = np.array(joints.velocities(iteration))[
            self.joints_map.indices[ControlType.TORQUE]
        ]
        stiffness = self.gammas[ControlType.TORQUE]*(
            positions
            - self.joints_map.bias[ControlType.TORQUE]
        )
        damping = self.deltas[ControlType.TORQUE]*velocities
        torques = stiffness + damping
        for i, idx in enumerate(self.joints_map.indices[ControlType.TORQUE]):
            joints.array[iteration, idx, 8] = torques[i]
            joints.array[iteration, idx, 10] = stiffness[i]
            joints.array[iteration, idx, 11] = damping[i]
        return dict(zip(self.joints[ControlType.TORQUE], torques))

    def torques(self, iteration, time, timestep):
        """Torques"""
        return self.torque_equation(iteration, time, timestep)


class JointsMap:
    """Joints map"""

    def __init__(self, joints, joints_names, animat_options, control_types):
        super().__init__()
        self.indices = [
            np.array([
                joint_i
                for joint_i, joint in enumerate(joints_names)
                if joint in joints[control_type]
            ])
            for control_type in control_types
        ]
        gain_amplitudes = {
            joint.joint: joint.gain_amplitude
            for joint in animat_options.control.joints
        }
        self.gain_amplitude = [
            np.array([
                gain_amplitudes[joint]
                for joint in joints[control_type]
            ])
            for control_type in control_types
        ]
        offsets_bias = {
            joint.joint: joint.bias
            for joint in animat_options.control.joints
        }
        self.bias = [
            np.array([
                offsets_bias[joint]
                for joint in joints[control_type]
            ])
            for control_type in control_types
        ]
