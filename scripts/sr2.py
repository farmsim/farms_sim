#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time
import matplotlib.pyplot as plt
from farms_models.utils import get_sdf_path
from farms_amphibious.examples.simulation import simulation, profile
import farms_pylog as pylog


def main():
    """Main"""
    sdf = get_sdf_path(name='salamandra_robotica', version='2')
    pylog.info('Model SDF: {}'.format(sdf))
    profile(
        function=simulation,
        sdf=sdf,
        use_controller=False,
        water_arena=False
    )
    plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
