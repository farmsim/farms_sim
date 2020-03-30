"""Amphibious"""

import re
import numpy as np
import pybullet

from farms_bullet.model.animat import Animat
from farms_bullet.sensors.sensors import (
    Sensors,
    JointsStatesSensor,
    ContactsSensors
)
from farms_bullet.plugins.swimming import (
    viscous_forces,
    resistive_forces,
    swimming_motion,
    swimming_debug
)
import farms_pylog as pylog
from ..sensors.sensors import AmphibiousGPS


def links_ordering(text):
    """links ordering"""
    text = re.sub("version[0-9]_", "", text)
    text = re.sub("[a-z]", "", text)
    text = re.sub("_", "", text)
    text = int(text)
    return [text]


def initial_pose(identity, spawn_options, units):
    """Initial pose"""
    pybullet.resetBasePositionAndOrientation(
        identity,
        spawn_options.position,
        pybullet.getQuaternionFromEuler(
            spawn_options.orientation
        )
    )
    pybullet.resetBaseVelocity(
        objectUniqueId=identity,
        linearVelocity=np.array(spawn_options.velocity_lin)*units.velocity,
        angularVelocity=np.array(spawn_options.velocity_ang)/units.seconds
    )
    if (
            spawn_options.joints_positions is not None
            or spawn_options.joints_velocities is not None
    ):
        if spawn_options.joints_positions is None:
            spawn_options.joints_positions = np.zeros_like(
                spawn_options.joints_velocities
            )
        if spawn_options.joints_velocities is None:
            spawn_options.joints_velocities = np.zeros_like(
                spawn_options.joints_positions
            )
        for joint_i, (position, velocity) in enumerate(zip(
                spawn_options.joints_positions,
                spawn_options.joints_velocities
        )):
            pybullet.resetJointState(
                bodyUniqueId=identity,
                jointIndex=joint_i,
                targetValue=position,
                targetVelocity=velocity/units.seconds
            )


class Amphibious(Animat):
    """Amphibious animat"""

    def __init__(self, sdf, options, controller, timestep, iterations, units, **kwargs):
        super(Amphibious, self).__init__(options=options)
        self.sdf = sdf
        self.timestep = timestep
        self.n_iterations = iterations
        self.controller = controller
        self.data = (
            controller.animat_data
            if controller is not None
            else None
        )
        # Hydrodynamic forces
        self.masses = np.zeros(options.morphology.n_links())
        self.hydrodynamics = None
        # Sensors
        self.sensors = Sensors()
        # Physics
        self.units = units
        # Elements
        self.feet = kwargs.pop('feet', None)
        self.links_order = kwargs.pop('links', None)
        self.joints_order = kwargs.pop('joints', None)
        self.links_swimming = kwargs.pop('links_swimming', None)
        self.links_no_collisions = kwargs.pop('links_no_collisions', None)
        assert not kwargs, kwargs

    def links_identities(self):
        """Links"""
        return [self._links[link] for link in self.links_order]

    def joints_identities(self):
        """Joints"""
        return [self._joints[joint] for joint in self.joints_order]

    def spawn(self):
        """Spawn amphibious"""
        # Spawn
        self.spawn_sdf()
        # Sensors
        if self.data:
            self.add_sensors()
        # Body properties
        self.set_body_properties()
        # Debug
        self.hydrodynamics = [
            pybullet.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, 0, 0],
                lineColorRGB=[0, 0, 0],
                lineWidth=3*self.units.meters,
                lifeTime=0,
                parentObjectUniqueId=self.identity(),
                parentLinkIndex=i
            )
            for i in range(self.options.morphology.n_links_body())
        ]

    def spawn_sdf(self, verbose=False):
        """Spawn sdf"""
        if verbose:
            pylog.debug(self.sdf)
        self._identity = pybullet.loadSDF(
            self.sdf,
            useMaximalCoordinates=0,
            globalScaling=1
        )[0]
        initial_pose(self._identity, self.options.spawn, self.units)
        for joint_i in range(pybullet.getNumJoints(self.identity())):
            joint_info = pybullet.getJointInfo(self.identity(), joint_i)
            self._links[joint_info[12].decode("UTF-8")] = joint_i
            self._joints[joint_info[1].decode("UTF-8")] = joint_i
        if self.links_order is not None:
            for link in self.links_order:
                if link not in self._links:
                    self._links[link] = -1
                    break
            for link in self.links_order:
                assert link in self._links, 'Link {} not in {}'.format(
                    link,
                    self._links,
                )
        # pylog.debug('Joints found:\n{}'.format(self._joints.keys()))
        # # Set names
        # self._links['link_body_{}'.format(0)] = -1
        # for i in range(self.options.morphology.n_joints_body):
        #     self._links['link_body_{}'.format(i+1)] = self.joints_order[i]
        #     self._joints['joint_link_body_{}'.format(i)] = self.joints_order[i]
        # for leg_i in range(self.options.morphology.n_legs//2):
        #     for side in range(2):
        #         for joint_i in range(self.options.morphology.n_dof_legs):
        #             self._links[
        #                 self.convention.leglink2name(
        #                     leg_i=leg_i,
        #                     side_i=side,
        #                     joint_i=joint_i
        #                 )
        #             ] = self.joints_order[
        #                 self.convention.leglink2index(
        #                     leg_i=leg_i,
        #                     side_i=side,
        #                     joint_i=joint_i
        #                 )
        #             ]
        #             self._joints[
        #                 self.convention.legjoint2name(
        #                     leg_i=leg_i,
        #                     side_i=side,
        #                     joint_i=joint_i
        #                 )
        #             ] = self.joints_order[
        #                 self.convention.legjoint2index(
        #                     leg_i=leg_i,
        #                     side_i=side,
        #                     joint_i=joint_i
        #                 )
        #             ]
        if verbose:
            self.print_information()

    def add_sensors(self):
        """Add sensors"""
        # Links
        if self.links_order is not None:
            self.sensors.add({
                "links": AmphibiousGPS(
                    array=self.data.sensors.gps.array,
                    animat_id=self.identity(),
                    links=self.links_identities(),
                    options=self.options,
                    units=self.units
                )
            })
        # Joints
        if self.joints_order is not None:
            self.sensors.add({
                "joints": JointsStatesSensor(
                    self.data.sensors.proprioception.array,
                    self._identity,
                    self.joints_identities(),
                    self.units,
                    enable_ft=True
                )
            })
        # Contacts
        if self.links_order is not None and self.feet is not None:
            self.sensors.add({
                "contacts": ContactsSensors(
                    self.data.sensors.contacts.array,
                    [self._identity for _ in self.feet],
                    [self._links[foot] for foot in self.feet],
                    self.units.newtons
                )
            })

    def set_body_properties(self, verbose=False):
        """Set body properties"""
        # Masses
        n_links = pybullet.getNumJoints(self.identity())+1
        self.masses = np.zeros(n_links)
        for i in range(n_links):
            self.masses[i] = pybullet.getDynamicsInfo(self.identity(), i-1)[0]
        if verbose:
            pylog.debug('Body mass: {} [kg]'.format(np.sum(self.masses)))
        # Deactivate collisions
        if self.links_no_collisions is not None:
            self.set_collisions(self.links_no_collisions, group=0, mask=0)
        # Deactivate damping
        small = 0
        self.set_links_dynamics(
            self._links.keys(),
            linearDamping=small,
            angularDamping=small,
            jointDamping=small
        )
        # Friction
        self.set_links_dynamics(
            self._links.keys(),
            lateralFriction=0.5,
            spinningFriction=small,
            rollingFriction=small,
        )
        if self.feet is not None:
            self.set_links_dynamics(
                self.feet,
                lateralFriction=0.9,
                spinningFriction=small,
                rollingFriction=small,
                # contactStiffness=1e3,
                # contactDamping=1e6
            )

    def viscous_swimming_forces(self, iteration, water_surface, **kwargs):
        """Animat swimming physics"""
        viscous_forces(
            iteration,
            self.data.sensors.gps,
            self.data.sensors.hydrodynamics.array,
            [
                link_i
                for link_i in range(self.options.morphology.n_links_body())
                if (
                    self.data.sensors.gps.com_position(iteration, link_i)[2]
                    < water_surface
                )
            ],
            masses=self.masses,
            **kwargs
        )

    def resistive_swimming_forces(self, iteration, water_surface, **kwargs):
        """Animat swimming physics"""
        resistive_forces(
            iteration,
            self.data.sensors.gps,
            self.data.sensors.hydrodynamics.array,
            [
                link_i
                for link_i in range(self.options.morphology.n_links_body())
                if (
                    self.data.sensors.gps.com_position(iteration, link_i)[2]
                    < water_surface
                )
            ],
            masses=self.masses,
            **kwargs
        )

    def apply_swimming_forces(
            self, iteration, water_surface, link_frame=True, debug=False
    ):
        """Animat swimming physics"""
        swimming_motion(
            iteration,
            self.data.sensors.hydrodynamics.array,
            self.identity(),
            [
                [self.links_order.index(name), self._links[name]]
                for name in self.links_swimming
                if (
                    self.data.sensors.gps.com_position(
                        iteration,
                        self.links_order.index(name)
                    )[2]
                    < water_surface
                )
            ],
            link_frame=link_frame,
            units=self.units
        )
        if debug:
            swimming_debug(
                iteration,
                self.data.sensors.gps,
                [
                    [self.links_order.index(name), self._links[name]]
                    for name in self.links_swimming
                ]
            )

    def draw_hydrodynamics(self, iteration):
        """Draw hydrodynamics forces"""
        for i, line in enumerate(self.hydrodynamics):
            force = self.data.sensors.hydrodynamics.array[iteration, i, :3]
            self.hydrodynamics[i] = pybullet.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=np.array(force),
                lineColorRGB=[0, 0, 1],
                lineWidth=7*self.units.meters,
                parentObjectUniqueId=self.identity(),
                parentLinkIndex=i-1,
                replaceItemUniqueId=line
            )
