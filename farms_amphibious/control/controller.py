"""Network controller"""


from typing import Dict, List, Tuple, Callable, Union
import numpy as np
from farms_data.amphibious.data import AmphibiousData
from farms_bullet.model.control import ModelController, ControlType
from ..model.options import AmphibiousOptions
from .drive import DescendingDrive
from .network import NetworkODE
from .ekeberg import EkebergMuscleCy


class AmphibiousController(ModelController):
    """Amphibious network"""

    def __init__(
            self,
            joints: List[str],
            animat_options: AmphibiousOptions,
            animat_data: AmphibiousData,
            drive: DescendingDrive = None,
    ):
        joints_control_types: Dict[str, ControlType] = {
            joint.joint: joint.control_type
            for joint in animat_options.control.joints
        }
        super().__init__(
            joints=ModelController.joints_from_control_types(
                joints_names=joints,
                joints_control_types=joints_control_types,
            ),
            max_torques=ModelController.max_torques_from_control_types(
                joints_names=joints,
                max_torques={
                    joint.joint: joint.max_torque
                    for joint in animat_options.control.joints
                },
                joints_control_types=joints_control_types,
            ),
        )
        self.network: NetworkODE = NetworkODE(animat_data)
        self.animat_data: AmphibiousData = animat_data
        self.drive: Union[DescendingDrive, None] = drive

        # joints
        self.joints_map: JointsMap = JointsMap(
            joints=self.joints,
            joints_names=joints,
            animat_options=animat_options,
        )

        # Equations
        self.equations: List[List[Callable]] = [[], [], []]

        # Muscles
        self.ekeberg_muscles = None
        self.muscles_map: MusclesMap = MusclesMap(
            joints=self.joints,
            animat_options=animat_options,
            animat_data=animat_data,
        )

        ## Equations handling

        # Position control
        if self.joints[ControlType.POSITION]:
            self.equations[ControlType.POSITION] = [self.positions_network]

        # Torque control
        if self.joints[ControlType.TORQUE]:

            torque_equation = animat_options.control.torque_equation
            self.equations[ControlType.TORQUE] = [{
                'ekeberg_muscle': self.ekeberg_muscle,
                'ekeberg_muscle_explicit': self.ekeberg_muscle_explicit,
            }[torque_equation]]

            if torque_equation in (
                    'ekeberg_muscle',
                    'ekeberg_muscle_explicit',
            ):

                if torque_equation == 'ekeberg_muscle':
                    # Velocity (damper)
                    self.joints[ControlType.VELOCITY] = self.joints[ControlType.TORQUE]
                    self.equations[ControlType.VELOCITY] = [
                        self.velocities_ekeberg_damper,
                    ]
                    self.max_torques[ControlType.VELOCITY] = np.full(
                        len(self.max_torques[ControlType.TORQUE]),
                        np.inf,
                    )
                    self.dtv = np.zeros_like(self.joints_map.indices[ControlType.TORQUE])
                    self.dpg = np.zeros_like(self.joints_map.indices[ControlType.TORQUE])
                    self.dvg = np.ones_like(self.joints_map.indices[ControlType.TORQUE])
                indices = np.array(self.joints_map.indices[ControlType.TORQUE], dtype=np.uintc)
                self.ekeberg_muscles = EkebergMuscleCy(
                    network=self.network,
                    joints=self.animat_data.sensors.joints,
                    n_muscles=len(indices),
                    indices=indices,
                    parameters=np.array(self.muscles_map.arrays[ControlType.TORQUE], dtype=np.double),
                    groups=np.array(self.muscles_map.groups[ControlType.TORQUE], dtype=np.uintc),
                    gain=np.array(self.joints_map.transform_gain, dtype=np.double),
                    bias=np.array(self.joints_map.transform_bias, dtype=np.double),
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
        if self.ekeberg_muscles is not None:
            self.ekeberg_muscles.step(iteration)

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
            result = equation(iteration, time, timestep)
            if isinstance(result, tuple):
                return result
            output.update(result)
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
        outputs = self.network.outputs(iteration)
        indices = self.joints_map.indices[ControlType.POSITION]
        positions = (
            self.joints_map.transform_gain[indices]*(
                0.5*(
                    outputs[self.muscles_map.groups[ControlType.POSITION][0]]
                    - outputs[self.muscles_map.groups[ControlType.POSITION][1]]
                )
                + np.array(self.network.offsets(iteration))[
                    self.joints_map.indices[ControlType.POSITION]
                ]
            ) + self.joints_map.transform_bias[indices]
        )
        return dict(zip(self.joints[ControlType.POSITION], positions))

    def velocities_ekeberg_damper(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Tuple:
        """Position control to simulate damper properties in Ekeberg muscle"""
        damping = self.ekeberg_muscles.damping(iteration)
        self.max_torques[ControlType.VELOCITY][:] = np.abs(damping)
        return (
            dict(zip(self.joints[ControlType.TORQUE], self.dtv)),
            self.dpg,  # positionGains
            self.dvg,  # velocityGains
        )

    def ekeberg_muscle(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
        """Ekeberg muscle"""
        return dict(zip(
            self.joints[ControlType.TORQUE],
            self.ekeberg_muscles.torques_implicit(iteration),
        ))

    def ekeberg_muscle_explicit(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
        """Ekeberg muscle with explicit passive dynamics"""
        return dict(zip(
            self.joints[ControlType.TORQUE],
            self.ekeberg_muscles.torques(iteration),
        ))


class JointsMap:
    """Joints map"""

    def __init__(
            self,
            joints: List[List[str]],
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
            joint.joint: joint.transform.gain
            for joint in animat_options.control.joints
        }
        self.transform_gain = np.array([
            transform_gains[joint]
            for joint in joints_names
        ])
        transform_bias = {
            joint.joint: joint.transform.bias
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
            joints: List[List[str]],
            animat_options: AmphibiousOptions,
            animat_data: AmphibiousData,
    ):
        super().__init__()
        control_types = list(ControlType)

        # Muscles
        joint_muscle_map = {
            muscle.joint: muscle
            for muscle in animat_options.control.muscles
        }
        muscles = [
            [joint_muscle_map[joint] for joint in joints[control_type]]
            for control_type in control_types
        ]
        self.arrays = [
            np.array([
                [muscle.alpha, muscle.beta, muscle.gamma, muscle.delta]
                for muscle in muscles[control_type]
            ])
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
