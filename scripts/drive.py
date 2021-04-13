#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time

import numpy as np
import farms_pylog as pylog
from farms_bullet.model.control import ControlType
from farms_amphibious.experiment.simulation import (
    setup_from_clargs,
    simulation_setup,
    postprocessing_from_clargs,
)


def main():
    """Main"""

    # Setup simulation
    pylog.info('Creating simulation')
    clargs, sdf, animat_options, simulation_options, arena = setup_from_clargs()
    joints = {
        joint['joint']: joint
        for joint in animat_options['control']['joints']
    }
    muscles = {
        muscle['joint']: muscle
        for muscle in animat_options['control']['muscles']
    }
    animat_options['control']['torque_equation'] = 'passive'
    for name in ('j_tailBone', 'j_tail'):
        if name in joints:
            joints[name]['control_type'] = ControlType.TORQUE
            muscles[name]['alpha'] = 0
            muscles[name]['beta'] = 0
            muscles[name]['gamma'] = 0  # Spring stiffness
            muscles[name]['delta'] = 0  # Damping coeffiecient
    sim = simulation_setup(
        animat_sdf=sdf,
        animat_options=animat_options,
        simulation_options=simulation_options,
        arena=arena,
        use_controller=True,
    )

    # Animat data
    data = sim.animat().data
    links = data.sensors.links
    drives = data.network.drives

    # Run simulation
    pylog.info('Running simulation')
    n_iterations = sim.options['n_iterations']
    for iteration in sim.iterator(show_progress=sim.options.show_progress):

        # Get head position in world coordinates
        head_pos = np.array(links.urdf_position(
            iteration=iteration,
            link_i=0,
        ))
        # Get head orientation as quaternion
        head_ori = np.array(links.urdf_orientation(
            iteration=iteration,
            link_i=0,
        ))

        # Drives
        drives.array[min(iteration+1, n_iterations-1), 0] = np.clip(
            head_pos[0]+1,
            1,  # Min drive value
            3,  # Max drive value
        )
        drives.array[min(iteration+1, n_iterations-1), 1] = np.clip(
            0*head_pos[0],
            -0.1,  # Min drive value
            0.1,  # Max drive value
        )

        # Print information
        if not iteration % 100:
            pylog.info(
                (
                    'State at iteration {}:'
                    '\n  - Head position: {}'
                    '\n  - Head orientation: {}'
                    '\n  - Drives : [{}, {}]'
                ).format(
                    iteration,
                    head_pos,
                    head_ori,
                    drives.array[iteration+1, 0],
                    drives.array[iteration+1, 1],
                )
            )

    # Terminate simulation
    pylog.info('Terminating simulation')
    sim.end()

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
