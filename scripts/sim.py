#!/usr/bin/env python3
"""Run a simulation with FARMS"""

import os
import time

from farms_core import pylog
from farms_core.utils.profile import profile
from farms_core.simulation.options import Simulator, SimulationOptions
from farms_core.model.control import AnimatController
from farms_core.extensions.extensions import import_item
from farms_mujoco.simulation.task import TaskCallback
from farms_mujoco.simulation.simulation import Simulation as MuJoCoSimulation
from farms_mujoco.sensors.camera import CameraCallback, save_video
from farms_sim.utils.parse_args import sim_parse_args
from farms_sim.simulation import (
    setup_from_clargs,
    run_simulation,
    postprocessing_from_clargs,
)

ENGINE_BULLET = False
try:
    from farms_bullet.simulation.simulation import (
        AnimatSimulation as PybulletSimulation
    )
    ENGINE_BULLET = True
except ImportError:
    PybulletSimulation = None
    pybullet_simulation_kwargs = None


def main():
    """Main"""

    # Setup
    pylog.info('Loading options from clargs')
    clargs, exp_options, simulator  = setup_from_clargs()
    if simulator == Simulator.PYBULLET and not ENGINE_BULLET:
        raise ImportError('Pybullet or farms_bullet not installed')
    sim_options: SimulationOptions = exp_options.simulation
    animats_options = exp_options.animats

    # Data
    experiment_data_loader = import_item(exp_options.loaders.experiment_data)
    experiment_data = experiment_data_loader.from_options(exp_options)
    animats_data = experiment_data.animats

    # Controllers
    animats_controller_loaders = [
        import_item(animat_options.control.controller_loader)
        for animat_options in animats_options
    ]
    animats_controllers: list[AnimatController] = [
        animats_controller_loader.from_options(
            animat_data=animat_data,
            animat_options=animat_options,
            experiment_options=exp_options,
            animat_i=animat_i,
        )
        for (
                animat_i,
                (animats_controller_loader, animat_options, animat_data),
        ) in enumerate(zip(
                animats_controller_loaders,
                animats_options,
                animats_data,
        ))
    ]

    # Additional engine-specific options
    options = {}
    options['callbacks'] = []
    camera = None
    if simulator == Simulator.MUJOCO:
        if sim_options.video.path:
            camera = CameraCallback.from_options(exp_options)
            options['callbacks'] += [camera]
        if clargs.log_path:
            options['save_mjcf'] = os.path.join(
                clargs.log_path,
                'sim_init.mjcf',
            )
    elif simulator == Simulator.PYBULLET:
        options.update(
            pybullet_simulation_kwargs(
                animats_controllers=animats_controllers,
                animat_options=animats_options,
                sim_options=sim_options,
            )
        )

    # Extensions
    sim_extensions: list[TaskCallback] = [
        # Simulation extensions
        import_item(extension)
        for extension in sim_options.extensions
    ] + [
        # Animat extensions
        import_item(extension.loader).from_options(
            animat_i,
            animat_data,
            animat_options,
            exp_options,
            extension.config,
        )
        for animat_i, (animat_data, animat_options) in enumerate(zip(
                animats_data,
                animats_options,
        ))
        for extension in animat_options.extensions
    ]
    options['callbacks'] += sim_extensions

    # Simulation
    pylog.info('Creating simulation environment')
    sim: MuJoCoSimulation | PybulletSimulation = run_simulation(
        experiment_data=experiment_data,
        experiment_options=exp_options,
        animats_controllers=animats_controllers,
        simulator=simulator,
        **options,
    )

    # Post-processing
    pylog.info('Running post-processing')
    postprocessing_from_clargs(
        sim=sim,
        clargs=clargs,
        simulator=simulator,
    )
    if sim_options.video.path:
        save_video(
            camera=camera,
            video_path=os.path.join(
                sim_options.video.path,
                sim_options.video.name,
            ),
        )


def profile_simulation():
    """Profile simulation"""
    tic = time.time()
    clargs = sim_parse_args()
    profile(function=main, profile_filename=clargs.profile)
    pylog.info('Total simulation time: %s [s]', time.time() - tic)


if __name__ == '__main__':
    profile_simulation()
