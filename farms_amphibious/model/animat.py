"""Amphibious"""

import re
import pybullet

from farms_bullet.model.animat import Animat


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

        # Links masses
        link_mass_multiplier = {
            link.name: link.mass_multiplier
            for link in self.options.morphology.links
        }
        for link in self.links_map:
            if link in link_mass_multiplier:
                mass, *_ = pybullet.getDynamicsInfo(
                    bodyUniqueId=self.identity(),
                    linkIndex=self.links_map[link],
                )
                pybullet.changeDynamics(
                    bodyUniqueId=self.identity(),
                    linkIndex=self.links_map[link],
                    mass=link_mass_multiplier[link]*mass,
                )

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
