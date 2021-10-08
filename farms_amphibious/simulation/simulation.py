"""Amphibious simulation"""

from farms_bullet.model.animat import Animat
from farms_bullet.simulation.options import SimulationOptions
from farms_bullet.simulation.simulation import AnimatSimulation
from farms_bullet.model.model import SimulationModel, SimulationModels
from farms_bullet.interface.interface import Interfaces
from farms_bullet.swimming.drag import SwimmingHandler

from .interface import AmphibiousUserParameters


class AmphibiousSimulation(AnimatSimulation):
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
        self.swimming_handler = (
            SwimmingHandler(animat)
            if animat.options.physics.drag
            or animat.options.physics.sph
            else None
        )
        if isinstance(self.swimming_handler, SwimmingHandler):
            self.swimming_handler.set_hydrodynamics_scale(
                animat.options.scale_hydrodynamics
            )

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
