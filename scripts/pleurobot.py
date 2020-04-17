#!/usr/bin/env python3
"""Run pleurobot simulation with bullet"""

import os
import time
import matplotlib.pyplot as plt

import farms_pylog as pylog
from farms_amphibious.utils.utils import prompt
from farms_amphibious.utils.network import plot_networks_maps
from farms_amphibious.experiment.simulation import simulation, profile
from farms_amphibious.experiment.options import (
    get_pleurobot_options,
    amphibious_options,
)


def main():
    """Main"""

    sdf, animat_options = get_pleurobot_options()

    (
        simulation_options,
        arena_sdf,
    ) = amphibious_options(animat_options, use_water_arena=False)

    # Simulation
    sim = profile(
        function=simulation,
        animat_sdf=sdf,
        animat_options=animat_options,
        simulation_options=simulation_options,
        arena_sdf=arena_sdf,
        use_controller=True,
    )

    # Post-processing
    pylog.info('Simulation post-processing')
    log_path = 'pleurobot_results' if prompt('Save data', False) else ''
    video_name = os.path.join(log_path, 'simulation.mp4')
    if log_path and not os.path.isdir(log_path):
        os.mkdir(log_path)
    sim.postprocess(
        iteration=sim.iteration,
        log_path=log_path,
        plot=prompt('Show plots', False),
        video=video_name if sim.options.record else ''
    )

    # Plot network
    if prompt('Show connectivity maps', False):
        plot_networks_maps(animat_options.morphology, sim.animat().data)

    # Plot
    plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
