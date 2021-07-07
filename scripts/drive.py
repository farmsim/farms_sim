#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from simple_pid import PID
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from scipy.interpolate import splprep, splev

import farms_pylog as pylog
from farms_bullet.model.control import ControlType
from farms_amphibious.experiment.simulation import (
    setup_from_clargs,
    simulation_setup,
    postprocessing_from_clargs,
)

from Vector_fields2 import (
    orientation_to_reach,
    theoritic_traj,
)


def main():
    """Main"""

    # Setup simulation
    pylog.info('Creating simulation')
    clargs, sdf, animat_options, simulation_options, arena = setup_from_clargs()

    # Setup
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

    # Simulation
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
    # contact = data.sensors.contacts
    hydrodynamic = data.sensors.hydrodynamics
    drives = data.network.drives

    # Arena data
    arena_type = clargs.arena

    # PID parameters
    P = 1.3
    I = 0.1
    D = 0.5
    pid = PID(P, I, D, sample_time=clargs.timestep, output_limits=(-0.2, 0.2))

    # flag to enable trajectory with obstacle
    MIXED = False

    # Initial position
    traj, X, Y, U, V = theoritic_traj(
        clargs.position[0], clargs.position[1], MIXED)

    # logfile init
    mean_ori_t = []
    pos = []
    set_t = []
    control_t = []

    # Run simulation
    pylog.info('Running simulation')
    n_iterations = sim.options['n_iterations']
    max_joint = 5
    # init storage of joint orientation
    joint_orientation = np.zeros((max_joint, 1))
    for iteration in sim.iterator(show_progress=sim.options.show_progress):

        # Get hydrodynamics forces
        hydro = np.array([
            hydrodynamic.force(iteration=iteration, sensor_i=sensor_i)
            for sensor_i in range(14)
        ])

        # Get 1st joint position in world coordinates
        head_pos = np.array(links.urdf_position(
            iteration=iteration,
            link_i=0,
        ))

        pos.append(head_pos)

        # Get orientation as radian
        for joint_idx in np.arange(0, max_joint, 1):
            ori = np.array(links.urdf_orientation(
                iteration=iteration,
                link_i=joint_idx,
            ))
            phi = Rotation.from_quat(ori).as_euler('xyz')[2]
            joint_orientation[joint_idx] = phi

        # Mean orientation of the joints
        mean_ori = np.mean(joint_orientation)

        # set the orientation command for the PID
        pid.setpoint = orientation_to_reach(head_pos[0], head_pos[1], MIXED)
        error = ((pid.setpoint - mean_ori + np.pi) % (2*np.pi)) - np.pi
        control = -pid(pid.setpoint-error, dt=clargs.timestep)

        # fill logfile
        mean_ori_t.append(mean_ori)
        set_t.append(pid.setpoint)
        control_t.append(control)

        # Set forward drive
        if arena_type in ['water']:
            drives.array[min(iteration+1, n_iterations-1), 0] = 4.5
        elif arena_type in ['flat']:
            drives.array[min(iteration+1, n_iterations-1), 0] = 1.5
        elif np.count_nonzero(hydro[0]) < 3:
            drives.array[min(iteration+1, n_iterations-1), 0] = 1.5
        else:
            drives.array[min(iteration+1, n_iterations-1), 0] = 4.5

        # Set turn drive
        drives.array[min(iteration+1, n_iterations-1), 1] = control

        # Print information
        if not iteration % 100:
            pylog.info(
                (
                    'State at iteration {}:'
                    '\n  - Mid position: {}'
                    '\n  - Orientation set point: {}'
                    '\n  - Mean z orientation: {}'
                    '\n  - Drives : [{}, {}]'
                    '\n  - hydro forces: {}'
                    '\n  - count hydro: {}'
                ).format(
                    iteration,
                    head_pos,
                    np.degrees(pid.setpoint),
                    np.degrees(mean_ori),
                    drives.array[iteration+1, 0],
                    drives.array[iteration+1, 1],
                    hydro[0:8],
                    np.count_nonzero(hydro[0:8]),
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

    #   logfile generation
    mean_t = np.array(mean_ori_t)
    pos = np.array(pos)
    set_t = np.array(set_t)
    control_t = np.array(control_t)
    x = np.arange(0, clargs.duration, clargs.timestep)

    # interpolate the robot's trajectory
    tck, u = splprep([pos[:, 0], pos[:, 1]], s=0)
    pos1 = splev(u, tck)
    pos1 = np.array(pos1)

    # interpolate the robot's theoritic trajectory
    tck1, u1 = splprep([traj[:, 0], traj[:, 1]], s=0)
    traj1 = splev(u1, tck1)
    traj1 = np.array(traj1)
    _, _, score = assess_traj(pos1, traj1)
    pylog.info('Trajectory error is: {}'.format(score))

    # Plotting
    figs = plotting(
        t=x,
        pos=pos1,
        control=control_t,
        phi=mean_t, phi_c=set_t,
        traj=traj1,
        X=X, Y=Y, U=U, V=V,
        MIXED=MIXED,
    )

    if clargs.save:
        for fig_i, fig in enumerate(figs):
            fig.savefig(os.path.join(clargs.save, '{}.png'.format(fig_i)))
    if not clargs.headless:
        plt.show()


def plotting(t, pos, control, phi, phi_c, traj, X, Y, U, V, MIXED):
    """Plotting"""

    # Gain plot
    fig, ax = plt.subplots()
    ax.plot(t, control, label='control output')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('drive')
    ax.set_title("PID output")
    ax.legend()

    # Orientation plot
    fig2, ax2 = plt.subplots()
    ax2.plot(t, np.degrees(phi_c), label='Orientation setpoint')
    ax2.plot(t, np.degrees(phi), label='mean orientation')
    plt.grid(axis='x', color='0.95')
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('Angle [deg]')
    ax2.set_title("Orientation")
    ax2.legend()

    # Orientation plot
    fig3, ax3 = plt.subplots()
    plt.quiver(X, Y, U, V, angles="xy")
    ax3.plot(pos[0, :], pos[1, :], label='robot position')
    ax3.plot(pos[0, 0], pos[1, 0], 'x', label='Pleurobot initial position')
    ax3.plot(traj[0, :], traj[1, :], label='theoritic trajectory')
    if MIXED:
        circle1 = plt.Circle(
            [3.8, 0],
            0.3,
            color='r',
            fill=False,
            label='obstacle',
        )
        ax3.add_patch(circle1)

    ax3.set_xlabel('x coordinate [m]')
    ax3.set_ylabel('y coordinate [m]')
    ax3.set_title("Trajectory of the robot")
    ax3.legend()

    plt.gca().set_aspect('equal')

    return fig, fig2, fig3


def assess_traj(traj, t_traj):
    """Assess trajectory"""

    tree = KDTree(np.transpose(t_traj))
    dd, ii = tree.query(np.transpose(traj), k=1)

    return dd, ii, np.sum(dd)


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
