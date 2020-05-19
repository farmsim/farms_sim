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
    drag_forces,
    swimming_motion,
    swimming_debug
)
import farms_pylog as pylog
from farms_bullet.sensors.sensors import LinksStatesSensor
from ..utils.sdf import load_sdf, load_sdf_pybullet
from .options import SpawnLoader


def links_ordering(text):
    """links ordering"""
    text = re.sub('version[0-9]_', '', text)
    text = re.sub('[a-z]', '', text)
    text = re.sub('_', '', text)
    text = int(text)
    return [text]


def initial_pose(identity, joints, spawn_options, units):
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
            ).tolist()
        if spawn_options.joints_velocities is None:
            spawn_options.joints_velocities = np.zeros_like(
                spawn_options.joints_positions
            ).tolist()
        for joint, position, velocity in zip(
                joints,
                spawn_options.joints_positions,
                spawn_options.joints_velocities
        ):
            pybullet.resetJointState(
                bodyUniqueId=identity,
                jointIndex=joint,
                targetValue=position,
                targetVelocity=velocity/units.seconds
            )


class Amphibious(Animat):
    """Amphibious animat"""

    def __init__(self, sdf, options, controller, timestep, iterations, units):
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

    def links_identities(self):
        """Links"""
        return [self._links[link] for link in self.options.morphology.links_names()]

    def joints_identities(self):
        """Joints"""
        return [self._joints[joint] for joint in self.options.morphology.joints_names()]

    def spawn(self):
        """Spawn amphibious"""
        # Spawn
        use_pybullet_loader = self.options.spawn.loader == SpawnLoader.PYBULLET
        self.spawn_sdf(original=use_pybullet_loader)
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

    def spawn_sdf(self, verbose=False, original=False):
        """Spawn sdf"""
        if verbose:
            pylog.debug(self.sdf)
        if original:
            self._identity, self._links, self._joints = load_sdf_pybullet(
                sdf_path=self.sdf,
                morphology_links=self.options.morphology.links_names(),
            )
        else:
            self._identity, self._links, self._joints = load_sdf(
                sdf_path=self.sdf,
                force_concave=False,
                reset_control=False,
                verbose=True,
                mass_multiplier=self.options.morphology.mass_multiplier,
            )
        initial_pose(
            identity=self._identity,
            joints=self.joints_identities(),
            spawn_options=self.options.spawn,
            units=self.units,
        )
        if verbose:
            self.print_information()

    def add_sensors(self):
        """Add sensors"""
        # Links
        if self.options.control.sensors.gps:
            self.sensors.add({
                'gps': LinksStatesSensor(
                    array=self.data.sensors.gps.array,
                    animat_id=self.identity(),
                    links=[
                        self._links[link]
                        for link in self.options.control.sensors.gps
                    ],
                    units=self.units
                )
            })

        # Joints
        if self.options.control.sensors.joints:
            self.sensors.add({
                'joints': JointsStatesSensor(
                    array=self.data.sensors.proprioception.array,
                    model_id=self._identity,
                    joints=[
                        self._joints[joint]
                        for joint in self.options.control.sensors.joints
                    ],
                    units=self.units,
                    enable_ft=True
                )
            })

        # Contacts
        if self.options.control.sensors.contacts:
            self.sensors.add({
                'contacts': ContactsSensors(
                    array=self.data.sensors.contacts.array,
                    animat_ids=[
                        self._identity
                        for _ in self.options.control.sensors.contacts
                    ],
                    animat_links=[
                        self._links[foot]
                        for foot in self.options.control.sensors.contacts
                    ],
                    newtons=self.units.newtons
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
        if self.options.morphology.links_no_collisions is not None:
            self.set_collisions(
                self.options.morphology.links_no_collisions,
                group=0,
                mask=0
            )
        # Default link properties
        for link in self._links:
            # Default friction
            self.set_link_dynamics(
                link,
                lateralFriction=1,
                spinningFriction=0,
                rollingFriction=0,
            )
            # Default damping
            self.set_link_dynamics(
                link,
                linearDamping=0,
                angularDamping=0,
                jointDamping=0,
            )
        # Friction
        for link, lateral, spinning, rolling in zip(
                self.options.morphology.links_names(),
                self.options.morphology.links_friction_lateral,
                self.options.morphology.links_friction_spinning,
                self.options.morphology.links_friction_rolling,
        ):
            self.set_link_dynamics(
                link,
                lateralFriction=lateral,
                spinningFriction=spinning,
                rollingFriction=rolling,
            )

    def drag_swimming_forces(self, iteration, water_surface, **kwargs):
        """Animat swimming physics"""
        drag_forces(
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
            surface=water_surface,
            **kwargs
        )

    def apply_swimming_forces(
            self, iteration, water_surface, link_frame=True, debug=False
    ):
        """Animat swimming physics"""
        links = self.options.morphology.links_names()
        links_swimming = self.options.morphology.links_swimming
        swimming_motion(
            iteration,
            self.data.sensors.hydrodynamics.array,
            self.identity(),
            [
                [links.index(name), self._links[name]]
                for name in links_swimming
                if (
                    self.data.sensors.gps.com_position(
                        iteration,
                        links.index(name)
                    )[2] < water_surface
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
                    [links.index(name), self._links[name]]
                    for name in links_swimming
                ]
            )

    def draw_hydrodynamics(self, iteration, water_surface, margin=0.01):
        """Draw hydrodynamics forces"""
        gps = self.data.sensors.gps
        links = self.options.morphology.links_names()
        for i, (line, name) in enumerate(zip(
                self.hydrodynamics,
                self.options.morphology.links_swimming
        )):
            if (
                    gps.com_position(iteration, links.index(name))[2]
                    < water_surface + margin
            ):
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
