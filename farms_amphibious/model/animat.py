"""Amphibious"""

import re
import numpy as np
import pybullet

from farms_bullet.model.animat import Animat
from ..swimming.swimming import (
    drag_forces,
    swimming_motion,
    swimming_debug
)


def links_ordering(text):
    """links ordering"""
    text = re.sub('version[0-9]_', '', text)
    text = re.sub('[a-z]', '', text)
    text = re.sub('_', '', text)
    text = int(text)
    return [text]


class Amphibious(Animat):
    """Amphibious animat"""

    def __init__(self, sdf, options, controller, timestep, iterations, units):
        super(Amphibious, self).__init__(
            options=options,
            data=(
                controller.animat_data
                if controller is not None
                else None
            ),
            units=units,
        )
        self.sdf = sdf
        self.timestep = timestep
        self.n_iterations = iterations
        self.controller = controller

        # Hydrodynamic forces
        self.hydrodynamics_plot = None

    def spawn(self):
        """Spawn amphibious"""
        super().spawn()

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
