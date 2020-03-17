"""Sand"""

import pybullet
import numpy as np
from tqdm import tqdm


def main():
    """Main"""
    # Simulation parameters
    radius = 0.02
    size = 0.1
    height = 2
    duration = 10
    timestep = 1e-2
    # Simulation
    pybullet.connect(pybullet.GUI)  # , options="--opencl"
    pybullet.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=pybullet.createCollisionShape(
            pybullet.GEOM_PLANE,
            radius=0.5*0.1
        ),
        basePosition=[0, 0, 0]
    )
    sphere_id = pybullet.createCollisionShape(
        pybullet.GEOM_SPHERE,
        radius=radius
    )
    positions = [
        np.arange(-size, size, 2.1*radius),
        np.arange(-size, size, 2.1*radius),
        np.arange(5*radius, 5*radius+height, 2.1*radius),
    ]
    n_particles = 1
    print(np.shape(positions))
    for _positions in positions:
        n_particles *= len(_positions)
    print("Number of particles: {}".format(n_particles))
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
    pybullet.setGravity(0, 0, -9.81)
    particles_positions = [
        [pos_x, pos_y, pos_z]
        for pos_x in positions[0]
        for pos_y in positions[1]
        for pos_z in positions[2]
    ]
    for position in tqdm(particles_positions):
        sphere = pybullet.createMultiBody(
            baseMass=radius,
            baseCollisionShapeIndex=sphere_id,
            basePosition=np.array(position)+1e-1*(np.random.ranf(3))
        )
        pybullet.changeDynamics(
            sphere, -1,
            lateralFriction=0.7,
            spinningFriction=0.1,
            rollingFriction=0.1,
            linearDamping=0,
            angularDamping=0,
            contactStiffness=1e3,
            contactDamping=1e4,
        )
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
    pybullet.setTimeStep(1e-3)
    print("Running simulation")
    for current_time in tqdm(np.arange(0, duration, timestep)):
        pybullet.stepSimulation()


if __name__ == '__main__':
    main()
