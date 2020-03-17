"""Amphibious"""

import re
import numpy as np
import pybullet

from farms_bullet.plugins.swimming import (
    viscous_forces,
    resistive_forces,
    swimming_motion,
    swimming_debug
)
from farms_bullet.sensors.sensors import (
    Sensors,
    JointsStatesSensor,
    ContactsSensors
)

from farms_sdf.sdf import ModelSDF, Link, Joint

from ..animat import Animat
from .convention import AmphibiousConvention
from .animat_data import (
    AmphibiousOscillatorNetworkState,
    AmphibiousData
)
from .control import AmphibiousController
from .sensors import AmphibiousGPS


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
    # print(spawn_options.velocity_lin)
    # print(spawn_options.velocity_ang)
    # raise Exception
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

    def __init__(self, sdf, options, timestep, iterations, units):
        super(Amphibious, self).__init__(options=options)
        self.sdf = sdf
        self.timestep = timestep
        self.n_iterations = iterations
        self.convention = AmphibiousConvention(self.options)
        self.feet_names = [
            self.convention.leglink2name(
                leg_i=leg_i,
                side_i=side_i,
                joint_i=3
            )
            for leg_i in range(options.morphology.n_legs//2)
            for side_i in range(2)
        ]
        self.joints_order = None
        self.data = AmphibiousData.from_options(
            AmphibiousOscillatorNetworkState.default_state(iterations, options),
            options,
            iterations
        )
        # Hydrodynamic forces
        self.masses = np.zeros(options.morphology.n_links())
        self.hydrodynamics = None
        # Sensors
        self.sensors = Sensors()
        # Physics
        self.units = units
        self.scale = options.morphology.scale

    def spawn(self):
        """Spawn amphibious"""
        # Spawn
        self.spawn_sdf()
        # Controller
        self.setup_controller()
        # Sensors
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
                parentObjectUniqueId=self.identity,
                parentLinkIndex=i
            )
            for i in range(self.options.morphology.n_links_body())
        ]

    def spawn_sdf(self, verbose=False):
        """Spawn sdf"""
        print(self.sdf)
        self._identity = pybullet.loadSDF(
            self.sdf,
            useMaximalCoordinates=0,
            globalScaling=1
        )[0]
        initial_pose(self._identity, self.options.spawn, self.units)
        n_joints = pybullet.getNumJoints(self.identity)
        joints_names = [None for _ in range(n_joints)]
        joint_index = 0
        for joint_i in range(n_joints):
            joint_info = pybullet.getJointInfo(
                self.identity,
                joint_i
            )
            joints_names[joint_index] = joint_info[1].decode("UTF-8")
            joint_index += 1
        for joint_index in range(n_joints):
            joint_info = pybullet.getJointInfo(
                self.identity,
                joint_index
            )
            joints_names[joint_index] = joint_info[1].decode("UTF-8")
            # joint_index += 1
        joints_names_dict = {
            name: i
            for i, name in enumerate(joints_names)
        }
        self.joints_order = [
            joints_names_dict[name]
            for name in [
                self.convention.bodyjoint2name(i)
                for i in range(self.options.morphology.n_joints_body)
            ] + [
                self.convention.legjoint2name(leg_i, side_i, joint_i)
                for leg_i in range(self.options.morphology.n_legs//2)
                for side_i in range(2)
                for joint_i in range(self.options.morphology.n_dof_legs)
            ]
        ]
        # Set names
        self.links['link_body_{}'.format(0)] = -1
        for i in range(self.options.morphology.n_links_body()-1):
            self.links['link_body_{}'.format(i+1)] = self.joints_order[i]
            self.joints['joint_link_body_{}'.format(i)] = self.joints_order[i]
        for leg_i in range(self.options.morphology.n_legs//2):
            for side in range(2):
                for joint_i in range(self.options.morphology.n_dof_legs):
                    self.links[
                        self.convention.leglink2name(
                            leg_i=leg_i,
                            side_i=side,
                            joint_i=joint_i
                        )
                    ] = self.joints_order[
                        self.convention.leglink2index(
                            leg_i=leg_i,
                            side_i=side,
                            joint_i=joint_i
                        )
                    ]
                    self.joints[
                        self.convention.legjoint2name(
                            leg_i=leg_i,
                            side_i=side,
                            joint_i=joint_i
                        )
                    ] = self.joints_order[
                        self.convention.legjoint2index(
                            leg_i=leg_i,
                            side_i=side,
                            joint_i=joint_i
                        )
                    ]
        if verbose:
            self.print_information()

    def add_sensors(self):
        """Add sensors"""
        # Contacts
        self.sensors.add({
            "contacts": ContactsSensors(
                self.data.sensors.contacts.array,
                [self._identity for _ in self.feet_names],
                [self.links[foot] for foot in self.feet_names],
                self.units.newtons
            )
        })
        # Joints
        self.sensors.add({
            "joints": JointsStatesSensor(
                self.data.sensors.proprioception.array,
                self._identity,
                self.joints_order,
                self.units,
                enable_ft=True
            )
        })
        # Base link
        links = [
            [
                "link_body_{}".format(i),
                i,
                self.links["link_body_{}".format(i)]
            ]
            for i in range(self.options.morphology.n_links_body())
        ] + [
            [
                "link_leg_{}_{}_{}".format(leg_i, side, joint_i),
                # 12 + leg_i*2*4 + side_i*4 + joint_i,
                self.convention.leglink2index(
                    leg_i,
                    side_i,
                    joint_i
                )+1,
                self.links["link_leg_{}_{}_{}".format(
                    leg_i,
                    side,
                    joint_i,
                    n_body_joints=self.options.morphology.n_joints_body
                )]
            ]
            for leg_i in range(self.options.morphology.n_legs//2)
            for side_i, side in enumerate(["L", "R"])
            for joint_i in range(self.options.morphology.n_dof_legs)
        ]
        self.sensors.add({
            "links": AmphibiousGPS(
                array=self.data.sensors.gps.array,
                animat_id=self.identity,
                links=links,
                options=self.options,
                units=self.units
            )
        })

    def set_body_properties(self):
        """Set body properties"""
        # Masses
        for i in range(self.options.morphology.n_links()):
            self.masses[i] = pybullet.getDynamicsInfo(self.identity, i-1)[0]
        # Deactivate collisions
        links_no_collisions = [
            "link_body_{}".format(body_i)
            for body_i in range(0)
        ] + [
            "link_leg_{}_{}_{}".format(leg_i, side, joint_i)
            for leg_i in range(self.options.morphology.n_legs//2)
            for side in ["L", "R"]
            for joint_i in range(self.options.morphology.n_dof_legs-1)
        ]
        self.set_collisions(links_no_collisions, group=0, mask=0)
        # Deactivate damping
        links_no_damping = [
            "link_body_{}".format(body_i)
            for body_i in range(self.options.morphology.n_links_body())
        ] + [
            "link_leg_{}_{}_{}".format(leg_i, side, joint_i)
            for leg_i in range(self.options.morphology.n_legs//2)
            for side in ["L", "R"]
            for joint_i in range(self.options.morphology.n_dof_legs)
        ]
        small = 0
        self.set_links_dynamics(
            links_no_damping,
            linearDamping=small,
            angularDamping=small,
            jointDamping=small
        )
        # Friction
        self.set_links_dynamics(
            self.links,
            lateralFriction=0.5,
            spinningFriction=small,
            rollingFriction=small,
        )
        self.set_links_dynamics(
            self.feet_names,
            lateralFriction=0.7,
            spinningFriction=small,
            rollingFriction=small,
            # contactStiffness=1e3,
            # contactDamping=1e6
        )

    def setup_controller(self):
        """Setup controller"""
        if self.options.control.kinematics_file:
            self.controller = AmphibiousController.from_kinematics(
                self.identity,
                animat_options=self.options,
                animat_data=self.data,
                timestep=self.timestep,
                joints_order=self.joints_order,
                units=self.units
            )
        else:
            self.controller = AmphibiousController.from_data(
                self.identity,
                animat_options=self.options,
                animat_data=self.data,
                timestep=self.timestep,
                joints_order=self.joints_order,
                units=self.units
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
            self.identity,
            [
                [i, self.links["link_body_{}".format(i)]]
                for i in range(self.options.morphology.n_links_body())
                if (
                    self.data.sensors.gps.com_position(iteration, i)[2]
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
                    [i, self.links["link_body_{}".format(i)]]
                    for i in range(self.options.morphology.n_links_body())
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
                parentObjectUniqueId=self.identity,
                parentLinkIndex=i-1,
                replaceItemUniqueId=line
            )
