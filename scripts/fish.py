#!/usr/bin/env python3
"""Run fish simulation with bullet"""

import time
import matplotlib.pyplot as plt
import farms_pylog as pylog
from farms_bullet.utils.profile import profile
from farms_amphibious.experiment.fish import fish_simulation


def main():
    """Main"""
    profile(
        fish_simulation,
        fish_name='crescent_gunnel',
        fish_version='1',
    )
    plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
