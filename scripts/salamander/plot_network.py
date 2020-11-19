#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import farms_pylog as pylog
from farms_models.utils import get_simulation_data_path
from farms_data.amphibious.animat_data import AnimatData
from farms_amphibious.utils.network import plot_networks_maps
from farms_amphibious.experiment.options import get_salamander_options


def main(animat='salamander', version='v3'):
    """Main"""

    # Post-processing
    pylog.info('Simulation post-processing')
    animat_options = get_salamander_options()
    log_path = get_simulation_data_path(
        name=animat,
        version=version,
        simulation_name='default',
    )
    pylog.debug('Data saved, now loading back to check validity')
    data = AnimatData.from_file(
        filename=os.path.join(log_path, 'simulation.hdf5'),
        n_oscillators=54,
    )
    plot_networks_maps(animat_options.morphology, data)
    plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
