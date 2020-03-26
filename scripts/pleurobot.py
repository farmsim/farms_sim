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

    links = ['base_link', 'Head'] + [
        'link{}'.format(i+1)
        for i in range(27)
    ] + ['link_tailBone', 'link_tail']
    joints = [
        # 'base_link_fixedjoint',
        'jHead',
        'j2',
        'j3',
        'j4',
        'j5',
        'j6',
        'j_tailBone',
        'j7',
        'j8',
        'j9',
        'j10',
        'j11',
        'j_tail',
        'HindLimbLeft_Yaw',
        'HindLimbLeft_Pitch',
        'HindLimbLeft_Roll',
        'HindLimbLeft_Elbow',
        'HindLimbRight_Yaw',
        'HindLimbRight_Pitch',
        'HindLimbRight_Roll',
        'HindLimbRight_Elbow',
        'ForearmLeft_Yaw',
        'ForearmLeft_Pitch',
        'ForearmLeft_Roll',
        'ForearmLeft_Elbow',
        'ForearmRight_Yaw',
        'ForearmRight_Pitch',
        'ForearmRight_Roll',
        'ForearmRight_Elbow',
    ]
    feet = ['link{}'.format(i+1) for i in [14, 18, 22, 26]]
    links_no_collisions = [link for link in links if link not in feet]

    # Simulation
    profile(
        function=amphibious_simulation,
        animat_sdf=sdf,
        animat_options=animat_options,
        use_water_arena=False,
        use_controller=False,
        links=links,
        joints=joints,
        feet=feet,
        links_no_collisions=links_no_collisions,
    )
    plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
