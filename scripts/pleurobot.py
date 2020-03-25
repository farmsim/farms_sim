#!/usr/bin/env python3
"""Run pleurobot simulation with bullet"""

import time
import matplotlib.pyplot as plt
from farms_models.utils import get_sdf_path
from farms_amphibious.examples.simulation import (
    amphibious_simulation,
    profile,
    get_animat_options,
)
import farms_pylog as pylog


def main():
    """Main"""

    # Animat
    sdf = get_sdf_path(name='pleurobot', version='0')
    pylog.info('Model SDF: {}'.format(sdf))
    animat_options = get_animat_options(swimming=False)
    animat_options.morphology.n_legs = 4
    animat_options.morphology.n_dof_legs = 4
    animat_options.morphology.n_joints_body = 11

    links_order = ['base_link' 'Head'] + [
        'link{}' for i in range(19)
    ] + ['link_tailBone', 'link_tail']

    # Simulation
    profile(
        function=amphibious_simulation,
        animat_sdf=sdf,
        animat_options=animat_options,
        use_water_arena=False,
        use_controller=False,
        use_amphibious=False,
        # links_order=links_order
    )
    plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
