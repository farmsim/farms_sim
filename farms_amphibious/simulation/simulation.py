"""Amphibious simulation"""

import numpy as np

from farms_bullet.simulation.simulation import AnimatSimulation
from farms_bullet.model.model import SimulationModels
from farms_bullet.interface.interface import Interfaces
from farms_bullet.swimming.drag import SwimmingHandler

from .interface import AmphibiousUserParameters


def time_based_drive(iteration, n_iterations, interface):
    """Switch drive based on time"""
    drive_speed = interface.user_params.drive_speed()
    drive_speed.value = 1 + 4*iteration/n_iterations
    drive_speed.changed = True


def gps_based_drive(iteration, animat, interface):
    """Switch drive based on position"""
    distance = animat.data.sensors.gps.com_position(
        iteration=iteration-1 if iteration else 0,
        link_i=0
    )[0]
    swim_distance = 3
    value = interface.user_params.drive_speed().value
    if distance < -swim_distance:
        interface.user_params.drive_speed().value = 4 - (
            0.05*(swim_distance+distance)
        )
        if interface.user_params.drive_speed().value != value:
            interface.user_params.drive_speed().changed = True
    else:
        if interface.user_params.drive_speed().value != value:
            interface.user_params.drive_speed().changed = True



class AmphibiousSimulation(AnimatSimulation):
    """Amphibious simulation"""

    def __init__(self, simulation_options, animat, arena):
        super(AmphibiousSimulation, self).__init__(
            models=SimulationModels([animat, arena]),
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

    def update_controller(self, iteration, animat):
        """Update controller"""
        animat.controller.step(
            iteration=iteration,
            time=iteration*self.options.timestep,
            timestep=self.options.timestep,
        )

    def step(self, iteration):
        """Simulation step"""
        animat = self.animat()

        # Interface
        if not self.options.headless:

            # # Drive changes depending on simulation time
            # if animat.options.transition:
            #     time_based_drive(
            #         iteration,
            #         self.options.n_iterations,
            #         self.interface
            #     )

            # GPS based drive
            # gps_based_drive(iteration, self.animat, self.interface)

            # Update interface
            self.animat_interface(iteration)

        # Animat sensors
        animat.sensors.update(iteration)

        # Physics step
        if iteration < self.options.n_iterations-1:
            # Swimming
            if self.swimming_handler is not None:
                self.swimming_handler.step(iteration)

            # Update animat controller
            if animat.controller is not None:
                self.update_controller(iteration, animat)

    def animat_interface(self, iteration):
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
