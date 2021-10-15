"""Network controller"""


from typing import Dict, List, Tuple, Callable, Union
import numpy as np
from farms_data.amphibious.data import AmphibiousData
from farms_bullet.model.control import ModelController, ControlType
from ..model.options import AmphibiousOptions
from .drive import DescendingDrive
from .network import NetworkODE
from .position_muscle_cy import PositionMuscleCy
from .passive_cy import PassiveJointCy
from .ekeberg import EkebergMuscleCy


class AmphibiousController(ModelController):
    """Amphibious network"""

    def __init__(
            self,
            joints_names: List[str],
            animat_options: AmphibiousOptions,
            animat_data: AmphibiousData,
            drive: DescendingDrive = None,
    ):
        joints_control_types: Dict[str, List[ControlType]] = {
            joint.joint_name: joint.control_types
            for joint in animat_options.control.joints
        }
        super().__init__(
            joints_names=ModelController.joints_from_control_types(
                joints_names=joints_names,
                joints_control_types=joints_control_types,
            ),
            max_torques=ModelController.max_torques_from_control_types(
                joints_names=joints_names,
                max_torques={
                    joint.joint_name: joint.max_torque
                    for joint in animat_options.control.joints
                },
                joints_control_types=joints_control_types,
            ),
        )
        self.network: NetworkODE = NetworkODE(animat_data)
        self.animat_data: AmphibiousData = animat_data
        self.drive: Union[DescendingDrive, None] = drive

        # joints
        joints_map: JointsMap = JointsMap(
            joints=self.joints_names,
            joints_names=joints_names,
            animat_options=animat_options,
        )

        # Equations
        equations = {
            joint.joint_name: joint.equation
            for joint in animat_options.control.joints
        }
        self.equations: Tuple[List[Callable]] = [[], [], []]

        # Muscles
        muscle_map: MusclesMap = MusclesMap(
            joints=self.joints_names,
            animat_options=animat_options,
            animat_data=animat_data,
        )

        # Network to joints interface
        self.network2joints = {}

        # Position control
        if 'position' in equations.values():
            self.equations[ControlType.POSITION] += [self.positions_network]
            joints_indices = np.array([
                joint_i
                for joint_i, joint in enumerate(animat_options.control.joints)
                if joint.equation == 'position'
            ], dtype=np.uintc)
            joints_names = np.array(
                self.animat_data.sensors.joints.names,
                dtype=object,
            )[joints_indices].tolist()
            self.network2joints['position'] = PositionMuscleCy(
                joints_names=joints_names,
                joints_data=self.animat_data.sensors.joints,
                indices=joints_indices,
                network=self.network,
                parameters=np.array(muscle_map.arrays[ControlType.POSITION], dtype=np.double),
                osc_indices=np.array(muscle_map.osc_indices[ControlType.POSITION], dtype=np.uintc),
                gain=np.array(joints_map.transform_gain, dtype=np.double),
                bias=np.array(joints_map.transform_bias, dtype=np.double),
            )

        # Ekeberg muscle model control
        for torque_equation in ['ekeberg_muscle', 'ekeberg_muscle_explicit']:

            if torque_equation not in equations.values():
                continue

            joints_indices = np.array([
                joint_i
                for joint_i, joint in enumerate(animat_options.control.joints)
                if joint.equation == torque_equation
            ], dtype=np.uintc)
            joints_names = np.array(
                self.animat_data.sensors.joints.names,
                dtype=object,
            )[joints_indices].tolist()

            self.equations[ControlType.TORQUE] += [{
                'ekeberg_muscle': self.ekeberg_muscle,
                'ekeberg_muscle_explicit': self.ekeberg_muscle_explicit,
            }[torque_equation]]

            if torque_equation == 'ekeberg_muscle':
                # Velocity (damper)
                self.equations[ControlType.VELOCITY] += [
                    self.velocities_ekeberg_damper,
                ]
                self.velocity_indices_ekeberg = np.array([
                    joint_i
                    for joint_i, joint
                    in enumerate(self.joints_names[ControlType.VELOCITY])
                    if joint in equations
                    and equations[joint] == 'ekeberg_muscle'
                ], dtype=np.uintc)
                self.velocity_targets_ekeberg = np.zeros_like(
                    self.velocity_indices_ekeberg,
                    dtype=np.double,
                )

            self.network2joints[torque_equation] = EkebergMuscleCy(
                joints_names=joints_names,
                joints_data=self.animat_data.sensors.joints,
                indices=joints_indices,
                network=self.network,
                parameters=np.array(muscle_map.arrays[ControlType.TORQUE], dtype=np.double),
                osc_indices=np.array(muscle_map.osc_indices[ControlType.TORQUE], dtype=np.uintc),
                gain=np.array(joints_map.transform_gain, dtype=np.double),
                bias=np.array(joints_map.transform_bias, dtype=np.double),
            )

        # Passive joint control
        if 'passive' in equations.values():

            joints_indices = np.array([
                joint_i
                for joint_i, joint in enumerate(animat_options.control.joints)
                if joint.equation == 'passive'
            ], dtype=np.uintc)
            joints_names = np.array(
                self.animat_data.sensors.joints.names,
                dtype=object,
            )[joints_indices].tolist()

            self.equations[ControlType.TORQUE] += [self.passive]
            self.equations[ControlType.VELOCITY] += [
                self.velocities_passive_damper,
            ]
            self.velocity_indices_passive = np.array([
                joint_i
                for joint_i, joint
                in enumerate(self.joints_names[ControlType.VELOCITY])
                if joint in equations
                and equations[joint] == 'passive'
            ], dtype=np.uintc)
            self.velocity_targets_passive = np.zeros_like(
                self.velocity_indices_passive,
                dtype=np.double,
            )

            self.network2joints['passive'] = PassiveJointCy(
                stiffness_coefficients=np.array([
                    joint.passive.stiffness_coefficient
                    for joint in animat_options.control.joints
                    if joint.equation == 'passive'
                ], dtype=np.double),
                damping_coefficients=np.array([
                    joint.passive.damping_coefficient
                    for joint in animat_options.control.joints
                    if joint.equation == 'passive'
                ], dtype=np.double),
                friction_coefficients=np.array([
                    joint.passive.friction_coefficient
                    for joint in animat_options.control.joints
                    if joint.equation == 'passive'
                ], dtype=np.double),
                joints_names=joints_names,
                joints_data=self.animat_data.sensors.joints,
                indices=joints_indices,
                gain=np.array(joints_map.transform_gain, dtype=np.double),
                bias=np.array(joints_map.transform_bias, dtype=np.double),
            )

    def step(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ):
        """Control step"""
        if self.drive is not None:
            self.drive.step(iteration, time, timestep)
        self.network.step(iteration, time, timestep)
        for net2joints in self.network2joints.values():
            net2joints.step(iteration)

    def positions(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
        """Positions"""
        output = {}
        for equation in self.equations[ControlType.POSITION]:
            output.update(equation(iteration, time, timestep))
        return output

    def velocities(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Union[Dict[str, float], Tuple]:
        """Velocities"""
        output: Dict[str, float] = {}
        for equation in self.equations[ControlType.VELOCITY]:
            output.update(equation(iteration, time, timestep))
        return output

    def torques(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
        """Torques"""
        output = {}
        for equation in self.equations[ControlType.TORQUE]:
            output.update(equation(iteration, time, timestep))
        return output

    def positions_network(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
        """Positions network"""
        return dict(zip(
            self.network2joints['position'].joints_names,
            self.network2joints['position'].position_cmds(iteration),
        ))

    def velocities_ekeberg_damper(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
        """Position control to simulate damper properties in Ekeberg muscle"""
        self.max_torques[ControlType.VELOCITY][self.velocity_indices_ekeberg] = (
            np.abs(self.network2joints['ekeberg_muscle'].damping(iteration))
            + np.abs(self.network2joints['ekeberg_muscle'].friction(iteration))
        )
        return dict(zip(
            self.network2joints['ekeberg_muscle'].joints_names,
            self.velocity_targets_ekeberg,
        ))

    def ekeberg_muscle(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
        """Ekeberg muscle"""
        return dict(zip(
            self.network2joints['ekeberg_muscle'].joints_names,
            self.network2joints['ekeberg_muscle'].torques_implicit(iteration),
        ))

    def ekeberg_muscle_explicit(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
        """Ekeberg muscle with explicit passive dynamics"""
        return dict(zip(
            self.network2joints['ekeberg_muscle_explicit'].joints_names,
            self.network2joints['ekeberg_muscle_explicit'].torque_cmds(iteration),
        ))

    def velocities_passive_damper(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
        """Position control to simulate damper properties in Passive joint"""
        self.max_torques[ControlType.VELOCITY][self.velocity_indices_passive] = (
            np.abs(self.network2joints['passive'].damping(iteration))
            + np.abs(self.network2joints['passive'].friction(iteration))
        )
        return dict(zip(
            self.network2joints['passive'].joints_names,
            self.velocity_targets_passive,
        ))

    def passive(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
        """Passive joint"""
        return dict(zip(
            self.network2joints['passive'].joints_names,
            self.network2joints['passive'].stiffness(iteration),
        ))

    def passive_explicit(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
        """Passive joint with explicit passive dynamics"""
        return dict(zip(
            self.network2joints['passive'].joints_names,
            self.network2joints['passive'].torque_cmds(iteration),
        ))


class JointsMap:
    """Joints map"""

    def __init__(
            self,
            joints: Tuple[List[str]],
            joints_names: List[str],
            animat_options: AmphibiousOptions,
    ):
        super().__init__()
        control_types = list(ControlType)
        self.names = np.array(joints_names)
        self.indices = [  # Indices in animat data for specific control type
            np.array([
                joint_i
                for joint_i, joint in enumerate(joints_names)
                if joint in joints[control_type]
            ])
            for control_type in control_types
        ]
        transform_gains = {
            joint.joint_name: joint.transform.gain
            for joint in animat_options.control.joints
        }
        self.transform_gain = np.array([
            transform_gains[joint]
            for joint in joints_names
        ])
        transform_bias = {
            joint.joint_name: joint.transform.bias
            for joint in animat_options.control.joints
        }
        self.transform_bias = np.array([
            transform_bias[joint]
            for joint in joints_names
        ])


class MusclesMap:
    """Muscles map"""

    def __init__(
            self,
            joints: Tuple[List[str]],
            animat_options: AmphibiousOptions,
            animat_data: AmphibiousData,
    ):
        super().__init__()
        control_types = list(ControlType)
        joint_muscle_map = {
            muscle.joint_name: muscle
            for muscle in animat_options.control.muscles
        }
        muscles = [
            [
                joint_muscle_map[joint]
                for joint in joints[control_type]
                if joint in joint_muscle_map
            ]
            for control_type in control_types
        ]
        self.arrays = [
            np.array([
                [
                    muscle.alpha, muscle.beta,
                    muscle.gamma, muscle.delta, muscle.epsilon,
                ]
                for muscle in muscles[control_type]
            ])
            for control_type in control_types
        ]
        osc_names = animat_data.network.oscillators.names
        self.osc_indices = [
            [
                [osc_names.index(muscle.osc1) for muscle in muscles[control_type]],
                [osc_names.index(muscle.osc2) for muscle in muscles[control_type]],
            ]
            for control_type in control_types
        ]
