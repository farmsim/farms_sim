"""Amphibious simulation"""

from typing import Dict

import numpy as np
from imageio import imread

from farms_bullet.model.animat import Animat
from farms_bullet.interface.interface import Interfaces
from farms_bullet.model.model import SimulationModel, SimulationModels
from farms_bullet.simulation.simulation import (
    AnimatSimulation as AnimatPybulletSimulation
)
from farms_mujoco.simulation.options import SimulationOptions
from farms_mujoco.swimming.drag import SwimmingHandler

from .interface import AmphibiousUserParameters


def water_velocity_from_maps(position, water_maps):
    """Water velocity from maps"""
    vel = np.array([
        water_maps[png][tuple(
            (
                max(0, min(
                    water_maps[png].shape[index]-1,
                    round(water_maps[png].shape[index]*(
                        (
                            position[index]
                            - water_maps['pos_min'][index]
                        ) / (
                            water_maps['pos_max'][index]
                            - water_maps['pos_min'][index]
                        )
                    ))
                ))
            )
            for index in range(2)
        )]
        for png_i, png in enumerate(['vel_x', 'vel_y'])
    ], dtype=np.double)
    vel[1] *= -1
    return vel


class AmphibiousPybulletSimulation(AnimatPybulletSimulation):
    """Amphibious simulation"""

    def __init__(
            self,
            simulation_options: SimulationOptions,
            animat: Animat,
            arena: SimulationModel = None,
    ):
        super().__init__(
            models=SimulationModels(
                [animat, arena]
                if arena is not None
                else [animat]
            ),
            options=simulation_options,
            interface=Interfaces(
                user_params=AmphibiousUserParameters(
                    animat_options=animat.options,
                    simulation_options=simulation_options,
                )
            )
        )

        # Swimming handling
        self.water_maps: Dict = {}
        self.constant_velocity: bool = True
        self.swimming_handler: SwimmingHandler = (
            SwimmingHandler(animat)
            if animat.options.physics.drag
            or animat.options.physics.sph
            else None
        )
        if isinstance(self.swimming_handler, SwimmingHandler):
            self.swimming_handler.set_hydrodynamics_scale(
                animat.options.scale_hydrodynamics
            )
            self.constant_velocity: bool = (
                len(animat.options.physics.water_velocity) == 3
            )
            if not self.constant_velocity:
                water_velocity = animat.options.physics.water_velocity
                water_maps = animat.options.physics.water_maps
                pngs = [np.flipud(imread(water_maps[i])).T for i in range(2)]
                pngs_info = [np.iinfo(png.dtype) for png in pngs]
                vels = [
                    (
                        png.astype(np.double) - info.min
                    ) * (
                        water_velocity[3] - water_velocity[0]
                    ) / (
                        info.max - info.min
                    ) +  water_velocity[0]
                    for png, info in zip(pngs, pngs_info)
                ]
                self.water_maps = {
                    'pos_min': np.array(water_velocity[6:8]),
                    'pos_max': np.array(water_velocity[8:10]),
                    'vel_x': vels[0],
                    'vel_y': vels[1],
                }

    def update_controller(self, iteration: int):
        """Update controller"""
        self.animat().controller.step(
            iteration=iteration,
            time=iteration*self.options.timestep,
            timestep=self.options.timestep,
        )

    def step(self, iteration: int):
        """Simulation step"""
        animat = self.animat()

        # Interface
        if not self.options.headless:
            self.animat_interface(iteration)

        # Animat sensors
        animat.sensors.update(iteration)

        # Swimming
        if self.swimming_handler is not None:
            if not self.constant_velocity:
                self.swimming_handler.set_water_velocity(
                    water_velocity_from_maps(
                        position=animat.data.sensors.links.urdf_position(
                            iteration=iteration,
                            link_i=0,
                        ),
                        water_maps = self.water_maps,
                    ),
                )
            self.swimming_handler.step(iteration)

        # Update animat controller
        if animat.controller is not None:
            self.update_controller(iteration)

    def animat_interface(self, iteration: int):
        """Animat interface"""
        animat = self.animat()

        # Drives
        if self.interface.user_params.drive_speed().changed:
            animat.data.network.drives.array[iteration, 0] = (
                self.interface.user_params.drive_speed().value
            )
            self.interface.user_params.drive_speed().changed = False

        # Turning
        if self.interface.user_params.drive_turn().changed:
            animat.data.network.drives.array[iteration, 1] = (
                self.interface.user_params.drive_turn().value
            )
            self.interface.user_params.drive_turn().changed = False
