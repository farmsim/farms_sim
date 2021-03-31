#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time

import farms_pylog as pylog
from farms_bullet.utils.profile import profile
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
    sim = profile(
        function=simulation,
        animat_sdf=sdf,
        animat_options=animat_options,
        simulation_options=simulation_options,
        arena=arena,
        use_controller=True,
        profile_filename=clargs.profile,
    )

    # Post-processing
    postprocessing_from_clargs(
        sim=sim,
        animat_options=animat_options,
        clargs=clargs,
    )


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
