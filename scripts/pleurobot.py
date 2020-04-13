#!/usr/bin/env python3
"""Run pleurobot simulation with bullet"""

import time
import numpy as np
import matplotlib.pyplot as plt
from farms_models.utils import get_sdf_path
from farms_amphibious.experiment.simulation import (
    simulation,
    amphibious_options,
    profile,
    get_animat_options,
)
import farms_pylog as pylog


def main():
    """Main"""

    # Animat
    sdf = get_sdf_path(name='pleurobot', version='0')
    pylog.info('Model SDF: {}'.format(sdf))

    # Amplitudes gains
    gain_amplitude = np.ones(13+4*4)  # np.ones(13+4*4)
    gain_amplitude[6] = 0
    gain_amplitude[12] = 0
    for leg_i in range(2):
        for side_i in range(2):
            mirror = (-1 if side_i else 1)
            mirror_full = (1 if leg_i else -1)*(1 if side_i else -1)
            gain_amplitude[13+2*leg_i*4+side_i*4+0] = mirror
            gain_amplitude[13+2*leg_i*4+side_i*4+1] = mirror
            gain_amplitude[13+2*leg_i*4+side_i*4+2] = -mirror
            gain_amplitude[13+2*leg_i*4+side_i*4+3] = mirror_full

    # Offsets gains
    gain_offset = np.ones(13+4*4)
    gain_offset[6] = 0
    gain_offset[12] = 0
    for leg_i in range(2):
        for side_i in range(2):
            mirror = (1 if side_i else -1)
            mirror_full = (1 if leg_i else -1)*(1 if side_i else -1)
            gain_offset[13+2*leg_i*4+side_i*4+0] = mirror
            gain_offset[13+2*leg_i*4+side_i*4+1] = mirror
            gain_offset[13+2*leg_i*4+side_i*4+2] = mirror_full
            gain_offset[13+2*leg_i*4+side_i*4+3] = mirror_full

    # Joints joints_offsets
    joints_offsets = np.zeros(13+4*4)
    for leg_i in range(2):
        for side_i in range(2):
            mirror = (1 if side_i else -1)
            mirror_full = (1 if leg_i else -1)*(1 if side_i else -1)
            joints_offsets[13+2*leg_i*4+side_i*4+0] = 0
            joints_offsets[13+2*leg_i*4+side_i*4+1] = 0
            joints_offsets[13+2*leg_i*4+side_i*4+2] = 0
            joints_offsets[13+2*leg_i*4+side_i*4+3] = mirror_full*np.pi/8

    # Animat options
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
    links_no_collisions = [
        link
        for link in links
        if link not in feet+['Head', 'link_tail']
    ]
    animat_options = get_animat_options(
        swimming=False,
        n_legs=4,
        n_dof_legs=4,
        n_joints_body=13,
        body_head_amplitude=0,
        body_tail_amplitude=0,
        body_stand_amplitude=0.3,
        # body_stand_shift=np.pi/4,
        # legs_amplitude=[0.8, np.pi/32, np.pi/4, np.pi/8],
        # legs_offsets_walking=[0, np.pi/32, 0, np.pi/8],
        # legs_offsets_swimming=[-2*np.pi/5, 0, 0, 0],
        body_stand_shift=np.pi/4,
        legs_amplitude=[np.pi/4, np.pi/8, np.pi/8, np.pi/8],
        legs_offsets_walking=[0, -np.pi/16, -np.pi/16, 0],
        legs_offsets_swimming=[2*np.pi/5, 0, 0, np.pi/2],
        gain_amplitude=gain_amplitude,
        gain_offset=gain_offset,
        joints_offsets=joints_offsets,
        weight_osc_body=1e0,
        weight_osc_legs_internal=3e1,
        weight_osc_legs_opposite=3e0,
        weight_osc_legs_following=3e0,
        weight_osc_legs2body=1e1,
        weight_sens_contact_i=0,
        weight_sens_contact_e=0,
        weight_sens_hydro_freq=0,
        weight_sens_hydro_amp=0,
        links=links,
        joints=joints,
        feet=feet,
        links_no_collisions=links_no_collisions,
    )

    (
        simulation_options,
        arena_sdf,
    ) = amphibious_options(animat_options, use_water_arena=False)

    # Simulation
    profile(
        function=simulation,
        animat_sdf=sdf,
        animat_options=animat_options,
        simulation_options=simulation_options,
        arena_sdf=arena_sdf,
        use_controller=True,
    )
    plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
