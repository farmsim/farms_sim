"""Amphibious simulation"""

import os
import time
import numpy as np

from farms_bullet.simulation.simulation import Simulation
from farms_bullet.simulation.simulator import real_time_handing
from farms_bullet.model.model import SimulationModels
from farms_bullet.interface.interface import Interfaces
import farms_pylog as pylog

from .interface import AmphibiousUserParameters


def swimming_step(iteration, animat):
    """Swimming step"""
    physics_options = animat.options.physics
    if physics_options.drag or physics_options.sph:
        water_surface = (
            np.inf
            if physics_options.sph
            else physics_options.water_surface
        )
        if physics_options.drag:
            animat.drag_swimming_forces(
                iteration,
                water_surface=water_surface,
                coefficients=physics_options.drag_coefficients,
                buoyancy=physics_options.buoyancy,
            )
        animat.apply_swimming_forces(
            iteration,
            water_surface=water_surface,
        )
        if animat.options.show_hydrodynamics:
            animat.draw_hydrodynamics(
                iteration,
                water_surface=water_surface,
            )


def time_based_drive(iteration, n_iterations, interface):
    """Switch drive based on time"""
    interface.user_params.drive_speed().value = (
        1+4*iteration/n_iterations
    )
    interface.user_params.drive_speed().changed = True


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



class AmphibiousSimulation(Simulation):
    """Amphibious simulation"""

    def __init__(self, simulation_options, animat, arena):
        super(AmphibiousSimulation, self).__init__(
            models=SimulationModels([animat, arena]),
            options=simulation_options
        )
        # Interface
        self.interface = Interfaces(
            user_params=AmphibiousUserParameters(
                animat_options=self.animat().options,
                simulation_options=simulation_options)
        )
        if not self.options.headless:
            self.interface.init_camera(
                target_identity=(
                    self.animat().identity()
                    if not self.options.free_camera
                    else None
                ),
                timestep=self.options.timestep,
                rotating_camera=self.options.rotating_camera,
                top_camera=self.options.top_camera,
                pitch=simulation_options.video_pitch,
                yaw=simulation_options.video_yaw,
            )

        if self.options.record:
            skips = int(2e-2/simulation_options.timestep)  # 50 fps
            self.interface.init_video(
                target_identity=self.animat().identity(),
                simulation_options=simulation_options,
                # fps=1./(skips*simulation_options.timestep),
                pitch=simulation_options.video_pitch,
                yaw=simulation_options.video_yaw,
                # skips=skips,
                motion_filter=10*skips*simulation_options.timestep,
                # distance=1,
                rotating_camera=self.options.rotating_camera,
                # top_camera=self.options.top_camera
            )
        # Real-time handling
        self.tic_rt = np.zeros(2)
        # Simulation state
        self.simulation_state = None
        self.save()

    def animat(self):
        """Salamander animat"""
        return self.models[0]

    def pre_step(self, iteration):
        """New step"""
        play = True
        # if not(iteration % 10000) and iteration > 0:
        #     pybullet.restoreState(self.simulation_state)
        #     state = self.animat().data.state
        #     state.array[self.animat().data.iteration] = (
        #         state.default_initial_state()
        #     )
        if not self.options.headless:
            play = self.interface.user_params.play().value
            if not iteration % int(0.1/self.options.timestep):
                self.interface.user_params.update()
            if not play:
                time.sleep(0.1)
                self.interface.user_params.update()
        return play

    def step(self, iteration):
        """Simulation step"""
        self.tic_rt[0] = time.time()
        # Interface
        if not self.options.headless:

            # Drive changes depending on simulation time
            if self.animat().options.transition:
                time_based_drive(
                    iteration,
                    self.options.n_iterations,
                    self.interface
                )

            # GPS based drive
            # gps_based_drive(iteration, self.animat, self.interface)

            # Update interface
            self.animat_interface(iteration)

        # Animat sensors
        self.animat().sensors.update(iteration)

        # Physics step
        if iteration < self.options.n_iterations-1:
            # Swimming
            swimming_step(iteration, self.animat())

            # Update animat controller
            if self.animat().controller is not None:
                self.animat().controller.step(
                    iteration=iteration,
                    time=iteration*self.options.timestep,
                    timestep=self.options.timestep,
                )

    def post_step(self, iteration):
        """Post step"""

        # Camera
        if not self.options.headless:
            self.interface.camera.update()
        if self.options.record:
            self.interface.video.record(iteration)

        # Real-time
        if not self.options.headless:
            self.tic_rt[1] = time.time()
            if (
                    not self.options.fast
                    and self.interface.user_params.rtl().value < 2.99
            ):
                real_time_handing(
                    self.options.timestep,
                    self.tic_rt,
                    rtl=self.interface.user_params.rtl().value
                )

    def animat_interface(self, iteration):
        """Animat interface"""
        animat = self.animat()

        # Camera zoom
        if self.interface.user_params.zoom().changed:
            self.interface.camera.set_zoom(
                self.interface.user_params.zoom().value
            )

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

    def postprocess(
            self,
            iteration,
            log_path='',
            plot=False,
            video='',
            **kwargs
    ):
        """Plot after simulation"""
        times = np.arange(
            0,
            self.options.timestep*self.options.n_iterations,
            self.options.timestep
        )[:iteration]

        # Log
        if log_path:
            pylog.info('Saving data to {}'.format(log_path))
            self.animat().data.to_file(
                os.path.join(
                    log_path,
                    'simulation.hdf5'
                ),
                iteration,
            )
            self.options.save(os.path.join(
                log_path,
                'simulation_options.yaml'
            ))
            self.animat().options.save(os.path.join(
                log_path,
                'animat_options.yaml'
            ))

        # Plot
        if plot:
            self.animat().data.plot(times)

        # Record video
        if video:
            self.interface.video.save(
                video,
                iteration=iteration,
                writer=kwargs.pop('writer', 'ffmpeg')
            )
