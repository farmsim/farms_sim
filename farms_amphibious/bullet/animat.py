"""Amphibious"""

import pybullet

from farms_data.model.control import ModelController
from farms_bullet.model.animat import Animat


class Amphibious(Animat):
    """Amphibious animat"""

    def __init__(
            self,
            sdf: str,
            controller: ModelController,
            timestep: float,
            iterations: int,
            **kwargs,
    ):
        super().__init__(
            data=controller.animat_data if controller is not None else None,
            **kwargs,
        )
        self.sdf: str = sdf
        self.timestep: float = timestep
        self.n_iterations: int = iterations
        self.controller: ModelController = controller
        self.hydrodynamics_plot: bool = None

    def spawn(self):
        """Spawn amphibious"""
        super().spawn()

        # Links masses
        link_mass_multiplier = {
            link.name: link.mass_multiplier
            for link in self.options.morphology.links
        }
        for link, index in self.links_map.items():
            if link in link_mass_multiplier:
                mass, _, torque, *_ = pybullet.getDynamicsInfo(
                    bodyUniqueId=self.identity(),
                    linkIndex=index,
                )
                pybullet.changeDynamics(
                    bodyUniqueId=self.identity(),
                    linkIndex=index,
                    mass=link_mass_multiplier[link]*mass,
                )
                pybullet.changeDynamics(
                    bodyUniqueId=self.identity(),
                    linkIndex=index,
                    localInertiaDiagonal=link_mass_multiplier[link]*torque,
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
        ] if self.options.show_hydrodynamics else []
