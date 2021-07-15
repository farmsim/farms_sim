#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time

import farms_pylog as pylog
from farms_bullet.utils.profile import profile
from farms_amphibious.utils.parse_args import parse_args
from farms_amphibious.experiment.simulation import (
    setup_from_clargs,
    simulation,
    postprocessing_from_clargs,
)


def main():
    """Main"""

    # Setup
    clargs, sdf, animat_options, simulation_options, arena = setup_from_clargs()

    # Simulation
    sim = simulation(
        animat_sdf=sdf,
        animat_options=animat_options,
        simulation_options=simulation_options,
        arena=arena,
        use_controller=True,
        drive_strategy=clargs.drive,
    )

    # Post-processing
    postprocessing_from_clargs(
        sim=sim,
        animat_options=animat_options,
        clargs=clargs,
    )


def profile_simulation():
    """Profile simulation"""
    clargs = parse_args()
    profile(function=main, profile_filename=clargs.profile)


if __name__ == '__main__':
    TIC = time.time()
    profile_simulation()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
