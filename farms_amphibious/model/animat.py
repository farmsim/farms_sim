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
import farms_pylog as pylog
from farms_bullet.sensors.sensors import LinksStatesSensor
from ..utils.sdf import load_sdf, load_sdf_pybullet
from ..swimming.swimming import (
    drag_forces,
    swimming_motion,
    swimming_debug
)
from .options import SpawnLoader


def links_ordering(text):
    """links ordering"""
    text = re.sub('version[0-9]_', '', text)
    text = re.sub('[a-z]', '', text)
    text = re.sub('_', '', text)
    text = int(text)
    return [text]


def initial_pose(identity, joints, joints_options, spawn_options, units):
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
    for joint, info in zip(joints, joints_options):
        pybullet.resetJointState(
            bodyUniqueId=identity,
            jointIndex=joint,
            targetValue=info.initial_position,
            targetVelocity=info.initial_velocity/units.seconds,
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
        self.masses = {}
        self.hydrodynamics_plot = None
        # Sensors
        self.sensors = Sensors()
        # Physics
        self.units = units

    def links_identities(self):
        """Links"""
        return [self.links_map[link] for link in self.options.morphology.links_names()]

    def joints_identities(self):
        """Joints"""
        return [self.joints_map[joint] for joint in self.options.morphology.joints_names()]

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
        self.hydrodynamics_plot = [
            [
                False,
                pybullet.addUserDebugLine(
                    lineFromXYZ=[0, 0, 0],
                    lineToXYZ=[0, 0, 0],
                    lineColorRGB=[0, 0, 0],
                    lineWidth=3*self.units.meters,
                    lifeTime=0,
                    parentObjectUniqueId=self.identity(),
                    parentLinkIndex=i
                )
            ]
            for i in range(self.data.sensors.hydrodynamics.array.shape[1])
        ]

    def spawn_sdf(self, verbose=False, original=False):
        """Spawn sdf"""
        if verbose:
            pylog.debug(self.sdf)
        if original:
            self._identity, self.links_map, self.joints_map = load_sdf_pybullet(
                sdf_path=self.sdf,
                morphology_links=self.options.morphology.links_names(),
            )
        else:
            self._identity, self.links_map, self.joints_map = load_sdf(
                sdf_path=self.sdf,
                force_concave=False,
                reset_control=False,
                verbose=True,
                links_options=self.options.morphology.links,
            )
        initial_pose(
            identity=self._identity,
            joints=self.joints_identities(),
            joints_options=self.options.morphology.joints,
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
                    model_id=self.identity(),
                    links=[
                        self.links_map[link]
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
                        self.joints_map[joint]
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
                    model_ids=[
                        self._identity
                        for _ in self.options.control.sensors.contacts
                    ],
                    model_links=[
                        self.links_map[foot]
                        for foot in self.options.control.sensors.contacts
                    ],
                    meters=self.units.meters,
                    newtons=self.units.newtons,
                )
            })

    def set_body_properties(self, verbose=False):
        """Set body properties"""
        # Masses
        for link in self.options.morphology.links:
            self.masses[link.name] = pybullet.getDynamicsInfo(
                self.identity(),
                self.links_map[link.name],
            )[0]
        if verbose:
            pylog.debug('Body mass: {} [kg]'.format(np.sum(self.masses.values())))
        # Deactivate collisions
        self.set_collisions(
            [
                link.name
                for link in self.options.morphology.links
                if not link.collisions
            ],
            group=0,
            mask=0
        )
        # Default dynamics
        for link in self.links_map:
            # Default friction
            self.set_link_dynamics(
                link,
                lateralFriction=0,
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
        # Model options dynamics
        for link in self.options.morphology.links:
            self.set_link_dynamics(
                link.name,
                **link.pybullet_dynamics,
            )
        for joint in self.options.morphology.joints:
            self.set_joint_dynamics(
                joint.name,
                **joint.pybullet_dynamics,
            )

    def drag_swimming_forces(self, iteration, links, **kwargs):
        """Animat swimming physics"""
        return drag_forces(
            iteration=iteration,
            data_gps=self.data.sensors.gps,
            data_hydrodynamics=self.data.sensors.hydrodynamics,
            links=links,
            masses=self.masses,
            **kwargs,
        )

    def apply_swimming_forces(self, iteration, links, link_frame, debug=False):
        """Animat swimming physics"""
        swimming_motion(
            iteration=iteration,
            data_hydrodynamics=self.data.sensors.hydrodynamics,
            model=self.identity(),
            links=links,
            links_map=self.links_map,
            link_frame=link_frame,
            units=self.units,
        )
        if debug:
            swimming_debug(
                iteration=iteration,
                data_gps=self.data.sensors.gps,
                links=links,
            )

    def draw_hydrodynamics(self, iteration, links):
        """Draw hydrodynamics forces"""
        active_links = [[hydro[0], False] for hydro in self.hydrodynamics_plot]
        for link in links:
            sensor_i = self.options.control.sensors.hydrodynamics.index(link.name)
            force = self.data.sensors.hydrodynamics.array[iteration, sensor_i, :3]
            self.hydrodynamics_plot[sensor_i] = True, pybullet.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=0.1*np.array(force),
                lineColorRGB=[0, 0, 1],
                lineWidth=7*self.units.meters,
                parentObjectUniqueId=self.identity(),
                parentLinkIndex=self.links_map[link.name],
                replaceItemUniqueId=self.hydrodynamics_plot[sensor_i][1],
            )
            active_links[sensor_i][1] = True
        for hydro_i, (old_active, new_active) in enumerate(active_links):
            if old_active and not new_active:
                self.hydrodynamics_plot[hydro_i] = (
                    False,
                    pybullet.addUserDebugLine(
                        lineFromXYZ=[0, 0, 0],
                        lineToXYZ=[0, 0, 0],
                        lineColorRGB=[0, 0, 1],
                        lineWidth=0,
                        parentObjectUniqueId=self.identity(),
                        parentLinkIndex=0,
                        replaceItemUniqueId=self.hydrodynamics_plot[hydro_i][1],
                    )
                )
