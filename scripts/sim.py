#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time
from typing import Union

from farms_core import pylog
from farms_core.utils.profile import profile
from farms_core.simulation.options import Simulator

from farms_mujoco.simulation.simulation import Simulation as MuJoCoSimulation
from farms_sim.utils.parse_args import sim_parse_args
from farms_sim.simulation import (
    setup_from_clargs,
    simulation,
    postprocessing_from_clargs,
)

ENGINE_BULLET = False
try:
    from farms_bullet.simulation.simulation import AnimatSimulation
    ENGINE_BULLET = True
except ImportError:
    AnimatSimulation = None


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
    sim: Union[MuJoCoSimulation, AnimatSimulation] = simulation(
        animat_options=animat_options,
        simulation_options=sim_options,
        arena_options=arena_options,
        use_controller=True,
        drive_config=animat_options.control.drive_config,
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
    clargs = sim_parse_args()
    profile(function=main, profile_filename=clargs.profile)
    pylog.info('Total simulation time: %s [s]', time.time() - tic)


if __name__ == '__main__':
    profile_simulation()
