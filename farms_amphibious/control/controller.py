"""Network controller"""

import numpy as np
from farms_bullet.model.control import ModelController, ControlType
from .network import NetworkODE
from .drive import OrientationFollower


class AmphibiousController(ModelController):
    """Amphibious network"""

    def __init__(self, joints, animat_options, animat_data, drive=None):
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
        self.drive = drive

        # joints
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
            'ekeberg_muscle_explicit': self.ekeberg_muscle_explicit,
        }[animat_options.control.torque_equation]

        # Muscles
        self.muscles_map = MusclesMap(
            joints=self.joints,
            animat_options=animat_options,
            animat_data=animat_data,
            control_types=self.control_types,
        )

        # Muscle constraints handling
        if (
                self.joints[ControlType.TORQUE]
                and self.torque_equation == self.ekeberg_muscle
        ):

            # Position
            self.joints[ControlType.POSITION] = self.joints[ControlType.TORQUE]
            self.joints_map.indices[ControlType.POSITION] = (
                self.joints_map.indices[ControlType.TORQUE]
            )
            self.max_torques[ControlType.POSITION] = np.full(
                len(self.max_torques[ControlType.TORQUE]),
                np.inf,
            )
            self.positions, self._positions = (
                self.positions_spring_damper,
                self.positions,
            )

            # # Velocity
            # self.joints[ControlType.VELOCITY] = self.joints[ControlType.TORQUE]
            # self.joints_map.indices[ControlType.VELOCITY] = (
            #     self.joints_map.indices[ControlType.TORQUE]
            # )
            # self.max_torques[ControlType.VELOCITY] = np.full(
            #     len(self.max_torques[ControlType.TORQUE]),
            #     np.inf,
            # )
            # self.velocities, self._velocities = (
            #     self.velocities_friction,
            #     self.velocities,
            # )

    def step(self, iteration, time, timestep):
        """Control step"""
        if self.drive is not None:
            self.drive.step(iteration, time, timestep)
        self.network.step(iteration, time, timestep)

    def positions(self, iteration, time, timestep):
        """Postions"""
        outputs = self.network.outputs(iteration)
        indices = self.joints_map.indices[ControlType.POSITION]
        positions = (
            self.joints_map.gain_amplitude[indices]*(
                0.5*(
                    outputs[self.muscles_map.groups[ControlType.POSITION][0]]
                    - outputs[self.muscles_map.groups[ControlType.POSITION][1]]
                )
                + np.array(self.network.offsets(iteration))[
                    self.joints_map.indices[ControlType.POSITION]
                ]
            ) + self.joints_map.bias[indices]
        )
        return dict(zip(self.joints[ControlType.POSITION], positions))

    def positions_spring_damper(self, iteration, time, timestep):
        """Postions"""
        joints = self.data.sensors.joints
        indices = self.joints_map.indices[ControlType.POSITION]
        joints_offsets = (
            self.joints_map.gain_amplitude[ControlType.TORQUE]
            *np.array(self.network.offsets(iteration))[indices]
            + self.joints_map.bias[ControlType.TORQUE]
        )
        positions = np.array(joints.positions(iteration))[indices]
        velocities = np.array(joints.velocities(iteration))[indices]
        betas = self.muscles_map.betas[ControlType.TORQUE]
        gammas = self.muscles_map.gammas[ControlType.TORQUE]
        deltas = self.muscles_map.deltas[ControlType.TORQUE]
        stiffness_coefficients = betas*gammas
        passive_stiffness = stiffness_coefficients*(joints_offsets - positions)
        damping = -deltas*velocities
        self.max_torques[ControlType.POSITION][:] = (
            np.abs(passive_stiffness + damping)
        )
        for i, idx in enumerate(indices):
            # joints.array[iteration, idx, 8] = torques[i]
            # joints.array[iteration, idx, 9] = active_torques[i] + active_stiffness[i]
            joints.array[iteration, idx, 10] = passive_stiffness[i]
            joints.array[iteration, idx, 11] = damping[i]
        return (
            dict(zip(self.joints[ControlType.POSITION], joints_offsets)),
            stiffness_coefficients*timestep,
            1+deltas,
        )

    def positions_spring(self, iteration, time, timestep):
        """Postions"""
        joints = self.animat_data.sensors.joints
        indices = self.joints_map.indices[ControlType.POSITION]
        joints_offsets = (
            self.joints_map.gain_amplitude[ControlType.TORQUE]
            *np.array(self.network.offsets(iteration))[indices]
            + self.joints_map.bias[ControlType.TORQUE]
        )
        positions = np.array(joints.positions(iteration))[indices]
        betas = self.muscles_map.betas[ControlType.TORQUE]
        gammas = self.muscles_map.gammas[ControlType.TORQUE]
        delta_phi = joints_offsets - positions
        passive_stiffness = np.abs(betas*gammas*delta_phi)
        n_joints = len(indices)
        self.max_torques[ControlType.POSITION][:] = passive_stiffness
        for i, idx in enumerate(indices):
            joints.array[iteration, idx, 10] = passive_stiffness[i]
        return dict(zip(self.joints[ControlType.POSITION], joints_offsets))

    def velocities_friction(self, iteration, time, timestep):
        """Postions"""
        joints = self.animat_data.sensors.joints
        indices = self.joints_map.indices[ControlType.VELOCITY]
        velocities = np.array(joints.velocities(iteration))[indices]
        n_joints = len(indices)
        max_torques = -self.muscles_map.deltas[ControlType.TORQUE]
        self.max_torques[ControlType.VELOCITY][:] = max_torques
        for i, idx in enumerate(indices):
            joints.array[iteration, idx, 11] = max_torques[i]
        return dict(zip(self.joints[ControlType.VELOCITY], np.zeros(n_joints)))

    def ekeberg_muscle_explicit(self, iteration, time, timestep):
        """Ekeberg muscle"""

        # Sensors
        joints = self.animat_data.sensors.joints
        indices = self.joints_map.indices[ControlType.TORQUE]
        positions = np.array(joints.positions(iteration))[indices]
        velocities = np.array(joints.velocities(iteration))[indices]

        # Neural activity
        neural_activity = self.network.outputs(iteration)

        # Joints offsets
        joints_offsets = (
            self.joints_map.gain_amplitude[ControlType.TORQUE]
            *np.array(self.network.offsets(iteration))[indices]
            + self.joints_map.bias[ControlType.TORQUE]
        )

        # Torques
        alphas = self.muscles_map.alphas[ControlType.TORQUE]
        betas = self.muscles_map.betas[ControlType.TORQUE]
        gammas = self.muscles_map.gammas[ControlType.TORQUE]
        deltas = self.muscles_map.deltas[ControlType.TORQUE]
        group0 = self.muscles_map.groups[ControlType.TORQUE][0]
        group1 = self.muscles_map.groups[ControlType.TORQUE][1]
        neural_diff = neural_activity[group0] - neural_activity[group1]
        neural_sum = neural_activity[group0] + neural_activity[group1]
        active_torques = neural_diff*alphas
        delta_phi = joints_offsets - positions
        active_stiffness = betas*neural_sum*delta_phi
        passive_stiffness = betas*gammas*delta_phi
        damping = -deltas*velocities

        # Final torques
        torques = np.clip(
            active_torques + active_stiffness + passive_stiffness + damping,
            a_min=-self.max_torques[ControlType.TORQUE],
            a_max=self.max_torques[ControlType.TORQUE],
        )
        for i, idx in enumerate(indices):
            joints.array[iteration, idx, 8] = torques[i]
            joints.array[iteration, idx, 9] = active_torques[i] + active_stiffness[i]
            joints.array[iteration, idx, 10] = passive_stiffness[i]
            joints.array[iteration, idx, 11] = damping[i]
        return dict(zip(self.joints[ControlType.TORQUE], torques))

    def ekeberg_muscle(self, iteration, time, timestep):
        """Ekeberg muscle"""

        # Sensors
        joints = self.data.sensors.joints
        indices = self.joints_map.indices[ControlType.TORQUE]
        positions = np.array(joints.positions(iteration))[indices]
        velocities = np.array(joints.velocities(iteration))[indices]

        # Neural activity
        neural_activity = self.network.outputs(iteration)

        # Joints offsets
        joints_offsets = (
            self.joints_map.gain_amplitude[ControlType.TORQUE]
            *np.array(self.network.offsets(iteration))[indices]
            + self.joints_map.bias[ControlType.TORQUE]
        )

        # Data
        alphas = self.muscles_map.alphas[ControlType.TORQUE]
        betas = self.muscles_map.betas[ControlType.TORQUE]
        gammas = self.muscles_map.gammas[ControlType.TORQUE]
        deltas = self.muscles_map.deltas[ControlType.TORQUE]
        group0 = self.muscles_map.groups[ControlType.TORQUE][0]
        group1 = self.muscles_map.groups[ControlType.TORQUE][1]
        neural_diff = neural_activity[group0] - neural_activity[group1]
        neural_sum = neural_activity[group0] + neural_activity[group1]
        delta_phi = joints_offsets - positions

        # Torques
        active_torques = neural_diff*alphas
        active_stiffness = betas*neural_sum*delta_phi
        passive_stiffness = betas*gammas*delta_phi
        damping = -deltas*velocities

        # Final torques
        torques = np.clip(
            active_torques + active_stiffness + passive_stiffness + damping,
            a_min=-self.max_torques[ControlType.TORQUE],
            a_max=self.max_torques[ControlType.TORQUE],
        )
        for i, idx in enumerate(indices):
            joints.array[iteration, idx, 8] = torques[i]
            joints.array[iteration, idx, 9] = active_torques[i] + active_stiffness[i]
            joints.array[iteration, idx, 10] = passive_stiffness[i]
            joints.array[iteration, idx, 11] = damping[i]
        return dict(zip(self.joints[ControlType.TORQUE], torques))

    def passive(self, iteration, time, timestep):
        """Passive joints"""
        joints = self.animat_data.sensors.joints
        indices = self.joints_map.indices[ControlType.TORQUE]
        positions = np.array(joints.positions(iteration))[indices]
        velocities = np.array(joints.velocities(iteration))[indices]
        stiffness = self.muscles_map.gammas[ControlType.TORQUE]*(
            positions
            - self.joints_map.bias[ControlType.TORQUE]
        )
        damping = self.muscles_map.deltas[ControlType.TORQUE]*velocities
        torques = stiffness + damping
        for i, idx in enumerate(indices):
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
        self.names = np.array(joints_names)
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
        self.gain_amplitude = np.array([
            gain_amplitudes[joint]
            for joint in joints_names
        ])
        offsets_bias = {
            joint.joint: joint.bias
            for joint in animat_options.control.joints
        }
        self.bias = np.array([
            offsets_bias[joint]
            for joint in joints_names
        ])


class MusclesMap:
    """Muscles map"""

    def __init__(self, joints, animat_options, animat_data, control_types):
        super().__init__()

        # Muscles
        joint_muscle_map = {
            muscle.joint: muscle
            for muscle in animat_options.control.muscles
        }
        muscles = [
            [joint_muscle_map[joint] for joint in joints[control_type]]
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
                animat_data.network.oscillators.names.index(muscle.osc1)
            )
            osc_map[muscle.osc2] = (
                animat_data.network.oscillators.names.index(muscle.osc2)
            )
        self.groups = [
            [
                [osc_map[muscle.osc1] for muscle in muscles[control_type]],
                [osc_map[muscle.osc2] for muscle in muscles[control_type]],
            ]
            for control_type in control_types
        ]
