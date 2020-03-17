#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time
import numpy as np
import matplotlib.pyplot as plt
from farms_models.utils import get_sdf_path
from farms_amphibious.model.animat import Amphibious
from farms_amphibious.model.simulation import AmphibiousSimulation
from farms_amphibious.model.animat_options import AmphibiousOptions
from farms_bullet.simulation.options import SimulationOptions
from farms_bullet.simulation.model import (
    SimulationModels,
    DescriptionFormatModel
)
import farms_pylog as pylog


def get_animat_options():
    """Get animat options - Should load a config file in the future"""
    scale = 1
    animat_options = AmphibiousOptions(
        # collect_gps=True,
        show_hydrodynamics=True,
        scale=scale
    )
    # animat_options.control.drives.forward = 4

    # Walking
    animat_options.spawn.position = [0, 0, scale*0.1]
    animat_options.spawn.orientation = [0, 0, 0]
    animat_options.physics.viscous = True
    animat_options.physics.buoyancy = True
    animat_options.physics.water_surface = True
    # Swiming
    # animat_options.spawn.position = [-10, 0, 0]
    # animat_options.spawn.orientation = [0, 0, np.pi]

    return animat_options


def get_simulation_options():
    """Get simulation options - Should load a config file in the future"""
    simulation_options = SimulationOptions.with_clargs()
    simulation_options.units.meters = 1
    simulation_options.units.seconds = 1e3
    simulation_options.units.kilograms = 1
    simulation_options.arena = "water"

    # Camera options
    simulation_options.video_yaw = 0
    simulation_options.video_pitch = -30
    simulation_options.video_distance = 1
    # simulation_options.video_name = (
    #     "transition_videos/swim2walk_y{}_p{}_d{}".format(
    #         simulation_options.video_yaw,
    #         simulation_options.video_pitch,
    #         simulation_options.video_distance,
    #     )
    # )

    return simulation_options


def flat_arena():
    """Flat arena"""
    return DescriptionFormatModel(
        path=get_sdf_path(
            name='arena_flat',
            version='v0',
        ),
        visual_options={
            'path': 'BIOROB2_blue.png',
            'rgbaColor': [1, 1, 1, 1],
            'specularColor': [1, 1, 1],
        }
    )


def water_arena(water_surface):
    """Water arena"""
    return SimulationModels(models=[
        DescriptionFormatModel(
            path=get_sdf_path(
                name='arena_ramp',
                version='angle_-10_texture',
            ),
            visual_options={
                'path': 'BIOROB2_blue.png',
                'rgbaColor': [1, 1, 1, 1],
                'specularColor': [1, 1, 1],
            }
        ),
        DescriptionFormatModel(
            path=get_sdf_path(
                name='arena_water',
                version='v0',
            ),
            spawn_options={
                'posObj': [0, 0, water_surface],
                'ornObj': [0, 0, 0, 1],
            }
        ),
    ])


def main():
    """Main"""

    # Get options
    show_progress = True
    animat_options = get_animat_options()
    simulation_options = get_simulation_options()

    # Creating arena
    # animat_options.physics.water_surface = None
    # animat_options.physics.viscous = False
    # animat_options.physics.sph = False
    # animat_options.physics.resistive = False
    # arena = flat_arena()
    animat_options.physics.water_surface = -0.1
    arena = water_arena(water_surface=animat_options.physics.water_surface)

    # Model sdf
    sdf = get_sdf_path(name='salamander', version='v1')
    # sdf = get_sdf_path(name='pleurobot', version='0')
    # sdf = get_sdf_path(name='salamandra_robotica', version='2')
    pylog.info('Model SDF: {}'.format(sdf))

    # Salamander options
    animat_options.morphology.n_dof_legs = 4
    animat_options.morphology.n_legs = 4

    # Animat sensors

    # Animat data
    # salamander_data = AmphibiousData.from_options(
    #     AmphibiousOscillatorNetworkState.default_state(iterations, options),
    #     options,
    #     iterations
    # )

    # Animat controller
    # controller = AmphibiousController.from_data(
    #     self.identity,
    #     animat_options=self.options,
    #     animat_data=self.data,
    #     timestep=self.timestep,
    #     joints_order=self.joints_order,
    #     units=self.units
    # )
    # controller = AmphibiousController.from_kinematics(
    #     self.identity,
    #     animat_options=self.options,
    #     animat_data=self.data,
    #     timestep=self.timestep,
    #     joints_order=self.joints_order,
    #     units=self.units
    # )

    # Creating animat
    salamander = Amphibious(
        sdf=sdf,
        options=animat_options,
        timestep=simulation_options.timestep,
        iterations=simulation_options.n_iterations,
        units=simulation_options.units,
    )

    # Setup simulation
    pylog.info("Creating simulation")
    sim = AmphibiousSimulation(
        simulation_options=simulation_options,
        animat=salamander,
        arena=arena,
    )

    # Run simulation
    pylog.info("Running simulation")
    # sim.run(show_progress=show_progress)
    # contacts = sim.models.animat.data.sensors.contacts
    for iteration in sim.iterator(show_progress=show_progress):
        # pylog.info(np.asarray(
        #     contacts.reaction(iteration, 0)
        # ))
        assert iteration >= 0

    # Analyse results
    pylog.info("Analysing simulation")
    sim.postprocess(
        iteration=sim.iteration,
        plot=simulation_options.plot,
        log_path=simulation_options.log_path,
        log_extension=simulation_options.log_extension,
        record=sim.options.record and not sim.options.headless
    )
    if simulation_options.log_path:
        np.save(
            simulation_options.log_path+"/hydrodynamics.npy",
            sim.models.animat.data.sensors.hydrodynamics.array
        )

    sim.end()

    # Show results
    plt.show()


def profile():
    """Profile with cProfile"""
    import cProfile
    import pstats
    cProfile.run("main()", "simulation.profile")
    pstat = pstats.Stats("simulation.profile")
    pstat.sort_stats('time').print_stats(30)
    pstat.sort_stats('cumtime').print_stats(30)


def pycall():
    """Profile with pycallgraph"""
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput
    with PyCallGraph(output=GraphvizOutput()):
        main()


if __name__ == '__main__':
    TIC = time.time()
    # main()
    profile()
    # pycall()
    pylog.info("Total simulation time: {} [s]".format(time.time() - TIC))
