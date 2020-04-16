"""Amphibious simulation"""

import os
import time
import numpy as np

from farms_bullet.simulation.simulation import Simulation
from farms_bullet.simulation.options import SimulationOptions
from farms_bullet.simulation.simulator import real_time_handing
from farms_bullet.model.model import SimulationModels
from farms_bullet.interface.interface import Interfaces
import farms_pylog as pylog

from ..model.animat import Amphibious
from ..model.options import AmphibiousOptions


def swimming_step(sim_step, animat):
    """Swimming step"""
    physics_options = animat.options.physics
    if (
            physics_options.resistive
            or physics_options.viscous
            or physics_options.sph
    ):
        water_surface = (
            np.inf
            if physics_options.sph
            else physics_options.water_surface
        )
        if physics_options.viscous:
            animat.viscous_swimming_forces(
                sim_step,
                water_surface=water_surface,
                coefficients=physics_options.viscous_coefficients,
                buoyancy=physics_options.buoyancy,
            )
        if physics_options.resistive:
            animat.resistive_swimming_forces(
                sim_step,
                water_surface=water_surface,
                coefficients=physics_options.resistive_coefficients,
                buoyancy=physics_options.buoyancy,
            )
        animat.apply_swimming_forces(
            sim_step,
            water_surface=water_surface,
        )
        if animat.options.show_hydrodynamics:
            animat.draw_hydrodynamics(
                sim_step,
                water_surface=water_surface,
            )


def time_based_drive(sim_step, n_iterations, interface):
    """Switch drive based on time"""
    interface.user_params.drive_speed().value = (
        1+4*sim_step/n_iterations
    )
    interface.user_params.drive_speed().changed = True


def gps_based_drive(sim_step, animat, interface):
    """Switch drive based on position"""
    distance = animat.data.sensors.gps.com_position(
        iteration=sim_step-1 if sim_step else 0,
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
        self.interface = Interfaces(int(10*1e-3/simulation_options.timestep))
        if not self.options.headless:
            self.interface.init_camera(
                target_identity=(
                    self.animat().identity()
                    if not self.options.free_camera
                    else None
                ),
                timestep=self.options.timestep,
                rotating_camera=self.options.rotating_camera,
                top_camera=self.options.top_camera
            )
            self.interface.init_debug(animat_options=self.animat().options)

        if self.options.record and not self.options.headless:
            skips = int(2e-2/simulation_options.timestep)  # 50 fps
            self.interface.init_video(
                target_identity=self.animat().identity(),
                simulation_options=simulation_options,
                fps=1./(skips*simulation_options.timestep),
                pitch=simulation_options.video_pitch,
                yaw=simulation_options.video_yaw,
                skips=skips,
                motion_filter=2*skips*simulation_options.timestep,
                distance=1,
                rotating_camera=self.options.rotating_camera,
                top_camera=self.options.top_camera
            )
        # Real-time handling
        self.tic_rt = np.zeros(2)
        # Simulation state
        self.simulation_state = None
        self.save()

    def animat(self):
        """Salamander animat"""
        return self.models[0]

    def pre_step(self, sim_step):
        """New step"""
        play = True
        # if not(sim_step % 10000) and sim_step > 0:
        #     pybullet.restoreState(self.simulation_state)
        #     state = self.animat().data.state
        #     state.array[self.animat().data.iteration] = (
        #         state.default_initial_state()
        #     )
        if not self.options.headless:
            play = self.interface.user_params.play().value
            if not sim_step % 100:
                self.interface.user_params.update()
            if not play:
                time.sleep(0.5)
                self.interface.user_params.update()
        return play

    def step(self, sim_step):
        """Simulation step"""
        self.tic_rt[0] = time.time()
        # Interface
        if not self.options.headless:

            # Drive changes depending on simulation time
            if self.animat().options.transition:
                time_based_drive(
                    sim_step,
                    self.options.n_iterations,
                    self.interface
                )

            # GPS based drive
            # gps_based_drive(sim_step, self.animat, self.interface)

            # Update interface
            self.animat_interface()

        # Animat sensors
        self.animat().sensors.update(sim_step)

        # Physics step
        if sim_step < self.options.n_iterations-1:
            # Swimming
            swimming_step(sim_step, self.animat())

            # Update animat controller
            if self.animat().controller is not None:
                self.animat().controller.control_step(
                    iteration=sim_step,
                    time=sim_step*self.options.timestep,
                    timestep=self.options.timestep,
                )

    def post_step(self, sim_step):
        """Post step"""

        # Camera
        if not self.options.headless:
            if self.options.record:
                self.interface.video.record(sim_step)
            # User camera
            self.interface.camera.update()

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

    def animat_interface(self):
        """Animat interface"""
        # Camera zoom
        if self.interface.user_params.zoom().changed:
            self.interface.camera.set_zoom(
                self.interface.user_params.zoom().value
            )

        # Body offset
        if self.interface.user_params.body_offset().changed:
            animat = self.animat()
            animat.options.control.network.joints.set_body_offsets(
                self.interface.user_params.body_offset().value,
                animat.options.morphology.n_joints_body
            )
            self.animat().controller.update(
                animat.options
            )
            self.interface.user_params.body_offset().changed = False

        # Drives
        if self.interface.user_params.drive_speed().changed:
            self.animat().options.control.drives.forward = (
                self.interface.user_params.drive_speed().value
            )
            self.animat().controller.update(
                self.animat().options
            )
            self.interface.user_params.drive_speed().changed = False

        # Turning
        if self.interface.user_params.drive_turn().changed:
            self.animat().options.control.drives.turning = (
                self.interface.user_params.drive_turn().value
            )
            self.animat().controller.update(
                self.animat().options
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


def main(simulation_options=None, animat_options=None):
    """Main"""

    # Parse command line arguments
    if not simulation_options:
        simulation_options = SimulationOptions.with_clargs()
    if not animat_options:
        animat_options = AmphibiousOptions()

    # Setup simulation
    pylog.debug('Creating simulation')
    sim = AmphibiousSimulation(
        simulation_options=simulation_options,
        animat=Amphibious(
            animat_options,
            simulation_options.timestep,
            simulation_options.n_iterations,
            simulation_options.units
        )
    )

    # Run simulation
    pylog.debug('Running simulation')
    sim.run()

    # Analyse results
    pylog.debug('Analysing simulation')
    sim.postprocess(
        iteration=sim.iteration,
        plot=simulation_options.plot,
        log_path=simulation_options.log_path,
        log_extension=simulation_options.log_extension,
        record=sim.options.record and not sim.options.headless
    )
    if simulation_options.log_path:
        np.save(
            simulation_options.log_path+'/hydrodynamics.npy',
            sim.models.animat.data.sensors.hydrodynamics.array
        )

    sim.end()


def main_parallel():
    """Simulation with multiprocessing"""
    from multiprocessing import Pool

    # Parse command line arguments
    sim_options = SimulationOptions.with_clargs()

    # Create Pool
    pool = Pool(2)

    # Run simulation
    pool.map(main, [sim_options, sim_options])
    pylog.debug('Done')


if __name__ == '__main__':
    # main_parallel()
    main()
