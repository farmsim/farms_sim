#!/usr/bin/env python3
"""Replay Pleurobot muscles data"""

import time
import numpy as np
import farms_pylog as pylog
from farms_data.amphibious.data import AmphibiousData
from farms_bullet.model.control import ModelController, ControlType
from farms_bullet.utils.profile import profile
from farms_amphibious.experiment.simulation import (
    setup_from_clargs,
    simulation,
    postprocessing_from_clargs,
)


class MuscleReplayController(ModelController):
    """Muscle replay controller"""

    def __init__(self, joints, animat_options, animat_data, muscles_csv, designators):
        super(MuscleReplayController, self).__init__(
            joints=joints,
            control_types={joint: ControlType.TORQUE for joint in joints},
            max_torques={joint: 1e2 for joint in joints},
        )
        self.muscles_csv = muscles_csv
        self.designators = designators
        self.animat_data = animat_data
        self.n_iterations = round(0.5*(self.muscles_csv.shape[0]-6))

        # Muscles
        self.muscles = {}
        for key in self.designators:
            if 'act' in key:
                continue
            self.muscles[key] = np.array([
                self.muscles_csv[self.designators.index(key), joint_i]
                for joint_i, joint in enumerate(joints)
            ])

    def torques(self, iteration, time, timestep):
        """Ekeberg muscle"""

        # Sensors
        joints = self.animat_data.sensors.joints
        positions = np.array(joints.positions(iteration))
        velocities = np.array(joints.velocities(iteration))

        # Neural activity
        assert self.designators[5+iteration] == 'f_act{}'.format(iteration)
        assert self.designators[5+iteration+self.n_iterations] == (
            'e_act{}'.format(iteration)
        )
        f_act = self.muscles_csv[5+iteration]
        e_act = self.muscles_csv[5+iteration+self.n_iterations]

        # Torques
        torques = (
            self.muscles['alpha']*(f_act - e_act)
            + self.muscles['beta']*(f_act + e_act + self.muscles['gamma'])
            *(positions - self.muscles['phi_r'])
            + self.muscles['delta']*velocities
        )
        return dict(zip(self.joints[ControlType.TORQUE], torques))


def main():
    """Main"""

    # Setup simulation
    pylog.info('Creating simulation')
    clargs, sdf, animat_options, simulation_options, arena = setup_from_clargs()

    # Options
    filename = '10000.csv'
    muscles_csv = np.genfromtxt(filename, delimiter=',')[1:, 1:]
    muscles_csv_str = np.genfromtxt(filename, delimiter=',', dtype=str)
    muscles_csv_joints = muscles_csv_str[0, 1:]
    muscles_csv_designators = list(muscles_csv_str[1:, 0])
    simulation_options.timestep = 1e-3
    simulation_options.n_iterations = round(0.5*(muscles_csv.shape[0]-6))

    # Joints order
    joints_names = animat_options.morphology.joints_names()
    assert all(muscles_csv_joints == joints_names), (
        '{} != {}'.format(muscles_csv_joints, joints_names)
    )

    # Animat data
    animat_data = AmphibiousData.from_options(
        animat_options.control,
        simulation_options.n_iterations,
        simulation_options.timestep,
    )

    # Controller
    controller = MuscleReplayController(
        joints=muscles_csv_joints,
        animat_options=animat_options,
        animat_data=animat_data,
        muscles_csv=muscles_csv,
        designators=muscles_csv_designators,
    )

    # Simulation
    sim = simulation(
        animat_sdf=sdf,
        animat_options=animat_options,
        simulation_options=simulation_options,
        arena=arena,
        animat_data=animat_data,
        animat_controller=controller,
    )

    # Post-processing
    postprocessing_from_clargs(
        sim=sim,
        animat_options=animat_options,
        clargs=clargs,
    )


if __name__ == '__main__':
    TIC = time.time()
    profile(main)
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
