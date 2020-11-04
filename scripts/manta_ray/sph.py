"""Manta ray swimming with SPH"""

import os
import sys
# from multiprocessing import Pool

import farms_pylog as pylog
from farms_models.utils import get_sdf_path
from farms_bullet.utils.profile import profile
from farms_sph.simulation import AmphibiousSimulation
from farms_amphibious.experiment.simulation import simulation_setup
from farms_amphibious.experiment.options import (
    get_simulation_options,
    get_flat_arena,
)

from manta import manta_options


class MantaRaySimulation(AmphibiousSimulation):
    """Rigid body to fluid coupling"""

    def __init__(
            self, duration=4, timestep=1e-3, spacing=1e-2, **kwargs
    ):
        # Kwargs
        density_solid = kwargs.pop('density', 1000)
        tank_size = kwargs.pop('tank_size', [0.5, 1, 0.2])  # [m]
        tank_position = [
            -0.5*1e3*tank_size[0],
            -1.0*1e3*tank_size[1]+1e3*0.1,
            -0.5*1e3*tank_size[2]+1e3*0.1,  # -1e3*0.2,  # +1e3*0.1,
        ]
        # tank_position = [
        #     -0.5*1e3*tank_size[0],
        #     -0.5*1e3*tank_size[1],
        #     -0*1e3*tank_size[2] - 1e3*0.4,
        # ]

        # Models
        manta_name = 'manta_ray'
        manta_version = '0'
        animat_sdf = get_sdf_path(
            name=manta_name,
            version=manta_version,
        )
        pylog.info('Model SDF: {}'.format(animat_sdf))
        sdf = get_sdf_path(name='manta_ray', version='0')
        arena = get_flat_arena()

        # Options
        animat_options = manta_options(sdf)
        sim_options = get_simulation_options(
            timestep=timestep,
            n_iterations=int(duration/timestep),
        )
        sim_options.gravity = [0, 0, 0]
        sim_options.fast = True
        sim_options.headless = True

        # Simulation
        sim = simulation_setup(
            animat_sdf=animat_sdf,
            animat_options=animat_options,
            simulation_options=sim_options,
            use_controller=True,
            arena=arena,
        )

        # Links swimming
        links_swimming = [
            link.name
            for link in sim.animat().options.morphology.links
            if link.swimming
        ]

        super(MantaRaySimulation, self).__init__(
            spacing=spacing,
            timestep=sim.options.timestep,
            duration=sim.options.duration(),
            simulation=sim,
            sdf_path=animat_sdf,
            links_swimming=links_swimming,
            factor=kwargs.pop('factor', 0.2),
            density=density_solid,
            tank_position=tank_position,
            tank_size=tank_size,
            **kwargs,
        )


def simulate(output='/tmp/manta_sph'):
    """Simulate"""
    app = MantaRaySimulation(
        duration=8,
        timestep=1e-4,
        spacing=1e-2,
        output_dir=output,
        density=1000,
        factor=0.5,
    )
    app.run()
    app.simulation.end()
    app.simulation.postprocess(
        iteration=app.simulation.iteration,
        log_path=app.log_path,
    )


def run(folder='/tmp/manta_ray_sph'):
    """Run simulation"""
    log_name = 'manta_ray'
    output = os.path.join(folder, log_name)
    if not os.path.isdir(output):
        os.makedirs(output)
    f_stdout, f_stderr = [
        os.path.join(output, 'std{}_{}.txt'.format(std, log_name))
        for std in ['out', 'err']
    ]
    with open(f_stdout, 'w+') as stdout, open(f_stderr, 'w+') as stderr:
        # sys.stdout = stdout
        # sys.stderr = stderr
        simulate(output=output)


def main(n_processes=None):
    """Main"""
    # kinematics = [None] + list(model_kinematics_files(
    #     name='manta',
    #     version='0',
    # ))
    # kwargs = {}
    # if n_processes is not None:
    #     kwargs['processes'] = n_processes
    # with Pool(**kwargs) as pool:
    #     pool.map(run, kinematics)
    run()


if __name__ == '__main__':
    # simulate(log=None)
    # main(n_processes=1)
    main()
    # profile(main)
    # debug()
