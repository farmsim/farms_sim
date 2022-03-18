#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time
from typing import Union

import farms_pylog as pylog
from farms_data.utils.profile import profile
from farms_data.simulation.options import Simulator

from farms_mujoco.simulation.simulation import Simulation as MuJoCoSimulation
from farms_amphibious.utils.parse_args import parse_args
from farms_amphibious.experiment.simulation import (
    setup_from_clargs,
    simulation,
    postprocessing_from_clargs,
)

ENGINE_BULLET = False
try:
    from farms_amphibious.bullet.simulation import AmphibiousPybulletSimulation
    ENGINE_BULLET = True
except ImportError as err:
    pylog.error(err)
    ENGINE_BULLET = False


def main():
    """Main"""

    # Setup
    pylog.info('Loading options from clargs')
    (
        clargs,
        animat_options,
        sim_options,
        arena_options,
    ) = setup_from_clargs()
    simulator = {
        'MUJOCO': Simulator.MUJOCO,
        'PYBULLET': Simulator.PYBULLET,
    }[clargs.simulator]

    if simulator == Simulator.PYBULLET and not ENGINE_BULLET:
        raise ImportError('Pybullet or farms_bullet not installed')

    # Simulation
    pylog.info('Creating simulation environment')
    sim: Union[MuJoCoSimulation, AmphibiousPybulletSimulation] = simulation(
        animat_options=animat_options,
        simulation_options=sim_options,
        arena_options=arena_options,
        use_controller=True,
        drive_config=clargs.drive_config,
        simulator=simulator,
    )

    # Post-processing
    pylog.info('Running post-processing')
    postprocessing_from_clargs(
        sim=sim,
        animat_options=animat_options,
        clargs=clargs,
        simulator=simulator,
    )


def profile_simulation():
    """Profile simulation"""
    tic = time.time()
    clargs = parse_args()
    profile(function=main, profile_filename=clargs.profile)
    pylog.info('Total simulation time: %s [s]', time.time() - tic)


if __name__ == '__main__':
    profile_simulation()
