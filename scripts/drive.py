#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import os
import time
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from simple_pid import PID
from scipy.stats import circmean
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

import farms_pylog as pylog
from farms_bullet.model.control import ControlType
from farms_amphibious.experiment.simulation import (
    setup_from_clargs,
    simulation_setup,
    postprocessing_from_clargs,
)


class PotentialMap(ABC):
    """Potential map"""

    @abstractmethod
    def heading(self, pos):
        """Heading"""
        raise NotImplementedError

    @staticmethod
    def limit_cycle():
        """Limit cycle"""
        return None

    def heading_cartesian(self, pos, radius=1):
        """Heading cartesian"""
        heading_complex = radius*np.exp(1j*self.heading(pos))
        return heading_complex.real, heading_complex.imag

    def mesh(self, lin_x, lin_y, radius=1):
        """Mesh"""
        dimensions = (len(lin_x), len(lin_y))
        vec_x, vec_y = np.meshgrid(lin_x, lin_y, indexing='ij')
        vec_u, vec_v = np.zeros(dimensions), np.zeros(dimensions)
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                vec_u[i, j], vec_v[i, j] = self.heading_cartesian(
                    pos=np.array([vec_x[i, j], vec_y[i, j]]),
                    radius=radius,
                )
        return vec_x, vec_y, vec_u, vec_v


class StraightLinePotentialMap(PotentialMap):
    """Straight line potential map"""

    def __init__(self, **kwargs):
        super().__init__()
        self.gain = kwargs.pop('gain', 1)
        self.origin = kwargs.pop('origin', np.zeros(2))
        self.theta = kwargs.pop('theta', np.pi/8)

    def heading(self, pos):
        """Heading"""
        pos_complex = complex(*(pos[:2] - self.origin))
        theta = np.angle(pos_complex)
        r_dot = self.gain*np.linalg.norm(pos_complex)*np.sin(self.theta-theta)
        return np.angle(
            r_dot*np.exp(1j*(self.theta+0.5*np.pi))
            + np.exp(1j*self.theta)
        )

    def limit_cycle(self):
        """Limit cycle"""
        vector_complex = np.exp(1j*self.theta)
        vector = np.array([vector_complex.real, vector_complex.imag])
        return np.array([self.origin + 1e3*vector, self.origin - 1e3*vector])


class CirclePotentialMap(PotentialMap):
    """Circle potential map"""

    def __init__(self, **kwargs):
        super().__init__()
        self.gain = kwargs.pop('gain', 1)
        self.origin = kwargs.pop('origin', np.zeros(2))
        self.radius = kwargs.pop('radius', 4)
        self.direction = kwargs.pop('direction', -1)

    def heading(self, pos):
        """Heading"""
        pos_complex = complex(*(pos[:2] - self.origin))
        r_dot = self.gain*(self.radius-np.abs(pos_complex))
        theta = np.angle(pos_complex)
        return np.angle(
            r_dot*np.exp(1j*theta)
            + np.exp(1j*(theta+0.5*np.sign(self.direction)*np.pi))
        )

    def limit_cycle(self):
        """Limit cycle"""
        vectors_complex = [
            self.radius*np.exp(1j*theta)
            for theta in np.linspace(0, 2*np.pi, 100)
        ]
        vectors = np.array([[vec.real, vec.imag] for vec in vectors_complex])
        return self.origin + vectors


class DescendingDrive:
    """Descending drive"""

    def __init__(self, drives):
        super().__init__()
        self._drives = drives
        self.n_iterations = np.shape(drives.array)[0]
        self.setpoints = np.zeros(self.n_iterations)
        self.control = np.zeros(self.n_iterations)

    def set_forward_drive(self, iteration, value):
        """Set forward drive"""
        self._drives.array[min(iteration+1, self.n_iterations-1), 0] = value

    def set_turn_drive(self, iteration, value):
        """Set turn drive"""
        self._drives.array[min(iteration+1, self.n_iterations-1), 1] = value


class OrientationFollower(DescendingDrive):
    """Descending drive to follow orientation"""

    def __init__(self, strategy, drives, hydrodynamics, timestep, **kwargs):
        super().__init__(drives=drives)
        self.strategy = strategy
        self.hydrodynamics = hydrodynamics
        self.timestep = timestep
        self.pid = PID(
            Kp=kwargs.pop('Kp', 1.3),
            Ki=kwargs.pop('Ki', 0.1),
            Kd=kwargs.pop('Kd', 0.5),
            sample_time=timestep,
            output_limits=kwargs.pop('output_limits', (-0.2, 0.2)),
        )

    def update_turn_command(self, pos):
        """Update command"""
        self.pid.setpoint = self.strategy.heading(pos)
        return self.pid.setpoint

    def update_turn_control(self, iteration, command, heading):
        """Update drive"""
        error = ((command - heading + np.pi)%(2*np.pi)) - np.pi
        self.set_turn_drive(
            iteration=iteration,
            value=-self.pid(command-error, dt=self.timestep),
        )
        return self._drives.array[iteration, 1]

    def update_foward_control(self, iteration, arena):
        """Update drive"""
        hydro = self.hydrodynamics.force(iteration=iteration, sensor_i=0)
        self.set_forward_drive(
            iteration=iteration,
            value=(
                4.5 if arena == 'water'
                else 1.5 if arena == 'flat'
                else 1.5 if np.count_nonzero(hydro) < 3
                else 4.5
            ),
        )

    def update(self, iteration, pos, heading, arena):
        """Update drive"""
        command = self.update_turn_command(pos=pos)
        control = self.update_turn_control(
            iteration=iteration,
            command=command,
            heading=heading,
        )
        self.update_foward_control(
            iteration=iteration,
            arena=arena,
        )
        self.setpoints[iteration] = command
        self.control[iteration] = control
        return command, control


def main():
    """Main"""

    # Setup simulation
    pylog.info('Creating simulation')
    clargs, sdf, animat_options, simulation_options, arena = setup_from_clargs()
    assert clargs.drive in ('line', 'circle')

    # Setup model
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
    hydrodynamics = data.sensors.hydrodynamics
    drives = data.network.drives

    # Descending drive
    drive = OrientationFollower(
        strategy={
            'line': StraightLinePotentialMap,
            'circle': CirclePotentialMap,
        }[clargs.drive](),
        drives=drives,
        hydrodynamics=hydrodynamics,
        timestep=clargs.timestep,
        Kp=0.2,
        Ki=0,
        Kd=0,
    )

    # Logfile init
    n_iterations = sim.options['n_iterations']
    mean_ori = np.zeros(n_iterations)
    head_pos = np.zeros([n_iterations, 3])

    # Run simulation
    pylog.info('Running simulation')
    max_joint = 5
    # init storage of joint orientation
    joint_orientation = np.zeros((max_joint, 1))
    for iteration in sim.iterator(show_progress=sim.options.show_progress):

        # Get hydrodynamics forces
        hydro = np.array([
            hydrodynamics.force(iteration=iteration, sensor_i=sensor_i)
            for sensor_i in range(14)
        ])

        # Get 1st joint position in world coordinates
        head_pos[iteration, :] = np.array(links.urdf_position(
            iteration=iteration,
            link_i=0,
        ))

        # Get orientation as radian
        for joint_idx in np.arange(max_joint):
            joint_orientation[joint_idx] = Rotation.from_quat(
                links.urdf_orientation(
                    iteration=iteration,
                    link_i=joint_idx,
                )
            ).as_euler('xyz')[2]

        # Circular mean orientation of the joints
        mean_ori[iteration] = circmean(
            samples=joint_orientation,
            low=-np.pi,
            high=np.pi,
        )

        # Set the orientation command for the PID
        drive.update(
            iteration=iteration,
            pos=head_pos[iteration, :],
            heading=mean_ori[iteration],
            arena=clargs.arena,
        )

        # Print information
        if not iteration % int(n_iterations/10):
            pylog.info(
                (
                    'State at iteration {}:'
                    '\n  - Mid position: {}'
                    '\n  - Orientation set point: {}'
                    '\n  - Mean z orientation: {}'
                    '\n  - Drives: [{}, {}]'
                    '\n  - hydro forces: {}'
                    '\n  - count hydro: {}'
                ).format(
                    iteration,
                    head_pos[iteration],
                    np.degrees(drive.pid.setpoint),
                    np.degrees(mean_ori[iteration]),
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

    # Plotting
    figs = plotting(
        times=np.arange(0, clargs.duration, clargs.timestep),
        pos=head_pos.T,
        drive=drive,
        phi=mean_ori,
    )

    if clargs.save:
        for fig_i, fig in enumerate(figs):
            fig.savefig(
                fname=os.path.join(clargs.save, '{}.pdf'.format(fig_i)),
                bbox_inches='tight',
            )
    if not clargs.headless:
        plt.show()


def plotting(times, pos, drive, phi):
    """Plotting"""

    # Gain plot
    fig, ax1 = plt.subplots()
    ax1.plot(times, drive.control, label='Control output')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Drive')
    ax1.set_title('PID output')
    ax1.legend()
    plt.grid(True)

    # Orientation plot
    fig2, ax2 = plt.subplots()
    ax2.plot(times, np.degrees(drive.setpoints), label='Orientation setpoint')
    ax2.plot(times, np.degrees(phi), label='Mean orientation')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Angle [deg]')
    ax2.set_title('Orientation')
    ax2.legend()
    plt.grid(True)

    # Position plot
    arrow_res = 0.5
    fig3, ax3 = plt.subplots()
    vec_x, vec_y, vec_u, vec_v = drive.strategy.mesh(
        lin_x=np.arange(
            start=np.floor(np.min(pos[:, 0])),
            stop=np.ceil(np.max(pos[:, 0]))+1,
            step=arrow_res,
        ),
        lin_y=np.arange(
            start=np.floor(np.min(pos[:, 1])),
            stop=np.ceil(np.max(pos[:, 1]))+1,
            step=arrow_res,
        ),
        radius=1.5*arrow_res,
    )
    plt.quiver(vec_x, vec_y, vec_u, vec_v, angles='xy')
    x_lim, y_lim = ax3.get_xlim(), ax3.get_ylim()
    limit_cycle = drive.strategy.limit_cycle()
    if limit_cycle is not None:
        ax3.plot(limit_cycle[:, 0], limit_cycle[:, 1], label='Limit trajectory')
    if pos is not None:
        ax3.plot(pos[0, :], pos[1, :], label='Robot trajectory')
        ax3.plot(pos[0, 0], pos[1, 0], 'x', label='Pleurobot initial position')
    ax3.set_xlim(x_lim)
    ax3.set_ylim(y_lim)
    # if MIXED:
    #     circle1 = plt.Circle(
    #         [3.8, 0],
    #         0.3,
    #         color='r',
    #         fill=False,
    #         label='obstacle',
    #     )
    #     ax3.add_patch(circle1)

    ax3.set_xlabel('X coordinate [m]')
    ax3.set_ylabel('Y coordinate [m]')
    ax3.set_title('Robot trajectory')
    ax3.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal')

    return fig, fig2, fig3


def assess_traj(traj, t_traj):
    """Assess trajectory"""
    tree = KDTree(np.transpose(t_traj))
    distances, indices = tree.query(np.transpose(traj), k=1)
    return distances, indices, np.sum(distances)


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
