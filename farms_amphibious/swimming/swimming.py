"""Swimming"""

import numpy as np
import pybullet

import farms_pylog as pylog


def drag_forces(
        iteration,
        data_gps,
        data_hydrodynamics,
        links,
        sensor_options,
        masses,
        gravity,
        use_buoyancy,
        surface,
):
    """Drag swimming"""
    links_swimming = []
    for link in links:
        sensor_i = sensor_options.gps.index(link.name)
        position = data_gps.com_position(iteration, sensor_i)
        if position[2] > surface:
            continue
        ori = data_gps.urdf_orientation(iteration, sensor_i)
        if not any(ori):
            continue
        links_swimming.append(link)
        lin_velocity = data_gps.com_lin_velocity(iteration, sensor_i)
        ang_velocity = data_gps.com_ang_velocity(iteration, sensor_i)

        # Compute velocity in local frame
        link_orientation_inv = np.array(
            pybullet.getMatrixFromQuaternion(ori)
        ).reshape([3, 3]).T
        link_velocity = np.dot(link_orientation_inv, lin_velocity)
        link_angular_velocity = np.dot(link_orientation_inv, ang_velocity)

        # Buoyancy forces
        buoyancy = np.dot(
            link_orientation_inv,
            [
                0, 0, -1000*masses[link.name]*gravity/link.density*min(
                    max(surface-position[2], 0)/link.height, 1
                )
            ]
        ) if use_buoyancy else np.zeros(3)

        # Drag forces
        sensor_i = sensor_options.hydrodynamics.index(link.name)
        coefficients = np.array(link.drag_coefficients)
        data_hydrodynamics[iteration, sensor_i, :3] = (
            np.sign(link_velocity)
            *coefficients[0]*link_velocity**2
            + buoyancy
        )
        data_hydrodynamics[iteration, sensor_i, 3:6] = (
            np.sign(link_angular_velocity)
            *coefficients[1]
            *link_angular_velocity**2
        )
    return links_swimming


def swimming_motion(
        iteration,
        data_hydrodynamics,
        model,
        links,
        links_map,
        sensor_options,
        link_frame,
        units,
):
    """Swimming motion"""
    for link in links:
        sensor_i = sensor_options.gps.index(link.name)
        pybullet.applyExternalForce(
            model,
            links_map[link.name],
            forceObj=(
                np.array(data_hydrodynamics[iteration, sensor_i, :3])
                *units.newtons
            ),
            posObj=[0, 0, 0],  # pybullet.getDynamicsInfo(model, link)[3]
            flags=pybullet.LINK_FRAME if link_frame else pybullet.WORLD_FRAME
        )
        pybullet.applyExternalTorque(
            model,
            links_map[link.name],
            torqueObj=(
                np.array(data_hydrodynamics[iteration, sensor_i, 3:6])
                *units.torques
            ),
            flags=pybullet.LINK_FRAME if link_frame else pybullet.WORLD_FRAME
        )


def swimming_debug(iteration, data_gps, links, sensor_options):
    """Swimming debug"""
    for link in links:
        sensor_i = sensor_options.gps.index(link.name)
        joint = np.array(data_gps.urdf_position(iteration, sensor_i))
        joint_ori = np.array(data_gps.urdf_orientation(iteration, sensor_i))
        # com_ori = np.array(data_gps.com_orientation(iteration, sensor_i))
        ori_joint = np.array(
            pybullet.getMatrixFromQuaternion(joint_ori)
        ).reshape([3, 3])
        # ori_com = np.array(
        #     pybullet.getMatrixFromQuaternion(com_ori)
        # ).reshape([3, 3])
        # ori = np.dot(ori_joint, ori_com)
        axis = 0.05
        offset_x = np.dot(ori_joint, np.array([axis, 0, 0]))
        offset_y = np.dot(ori_joint, np.array([0, axis, 0]))
        offset_z = np.dot(ori_joint, np.array([0, 0, axis]))
        pylog.debug('SPH position: {}'.format(np.array(joint)))
        for i, offset in enumerate([offset_x, offset_y, offset_z]):
            color = np.zeros(3)
            color[i] = 1
            pybullet.addUserDebugLine(
                joint,
                joint + offset,
                lineColorRGB=color,
                lineWidth=5,
                lifeTime=1,
            )