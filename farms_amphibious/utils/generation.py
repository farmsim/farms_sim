"""Generate amphibious model"""

import os
import shutil
from functools import partial
import numpy as np
from farms_sdf.sdf import ModelSDF, Link, Joint, Visual
from farms_models.utils import (
    get_sdf_path,
    get_model_path,
    create_new_model_from_farms_sdf,
)

from ..model.options import AmphibiousOptions
from ..model.convention import AmphibiousConvention


def generate_sdf(model_name, model_version, **kwargs):
    """Generate sdf"""

    # Arguments
    options = kwargs.pop('options')
    convention = kwargs.pop('convention')
    scale = kwargs.pop('scale', 0.2)
    use_2d = kwargs.pop('use_2d', False)
    color = kwargs.pop('color', [0.1, 0.7, 0.1, 1.0])
    eye_pos = kwargs.pop('eye_pos', [0.04, 0.03, 0.015])
    eye_radius = kwargs.pop('eye_radius', 0.01)
    leg_offset = kwargs.pop('leg_offset')
    leg_length = kwargs.pop('leg_length')
    leg_radius = kwargs.pop('leg_radius')
    legs_parents = kwargs.pop('legs_parents')
    assert not kwargs, kwargs

    # Augment parameters
    repeat = partial(np.repeat, repeats=convention.n_legs//2, axis=0)
    if np.ndim(leg_offset) == 1:
        leg_offset = repeat([leg_offset])

    # Links and joints
    links = [None for _ in range(options.morphology.n_links())]
    joints = [None for _ in range(options.morphology.n_joints())]
    max_velocity = 1e6
    max_torque = 1e6

    # Scale
    leg_radius *= scale
    leg_length *= scale
    leg_offset = scale*np.array(leg_offset)

    # Original
    body_sdf_path = get_sdf_path(name=model_name, version='body')
    original_model = ModelSDF.read(body_sdf_path)[0]

    # Body
    for i, link in enumerate(original_model.links):
        link.name = convention.bodylink2name(i)
        links[i] = link
        link.pose = np.array(link.pose, dtype=float).tolist()
        link.visuals[0].color = color
        link.visuals[0].ambient = link.visuals[0].color
        link.visuals[0].diffuse = link.visuals[0].color
        link.visuals[0].specular = link.visuals[0].color
        link.visuals[0].emissive = link.visuals[0].color
        if i == 0:
            for sign, name in [[-1, 'left'], [1, 'right']]:
                link.visuals.append(Visual.sphere(
                    'eye_{}'.format(name),
                    pose=[
                        scale*eye_pos[0], scale*sign*eye_pos[1], scale*eye_pos[2],
                        0.0, 0.0, 0.0,
                    ],
                    radius=scale*eye_radius,
                    color=[0.0, 0.0, 0.0, 1.0],
                ))
        for j in range(3):
            link.pose[j] = scale*link.pose[j]
        link.visuals[0].geometry.scale = [scale]*3
        link.collisions[0].geometry.scale = [scale]*3
        link.inertial.mass *= scale**3
        for j in range(3):
            link.inertial.pose[j] *= scale
        for j in range(6):
            link.inertial.inertias[j] *= scale**5

    angle_max = 2*np.pi/len(original_model.joints)
    for i, joint in enumerate(original_model.joints):
        joint.name = convention.bodyjoint2name(i)
        joint.pose = np.array(joint.pose, dtype=float).tolist()
        joint.parent = convention.bodylink2name(i)
        joint.child = convention.bodylink2name(i+1)
        joints[i] = joint
        for j in range(3):
            joint.pose[j] = scale*joint.pose[j]
        joint.axis.limits[0] = -angle_max
        joint.axis.limits[1] = +angle_max
        joint.axis.limits[2] = max_torque
        joint.axis.limits[3] = max_velocity

    # Leg links
    for leg_i in range(options.morphology.n_legs//2):
        for side_i in range(2):
            sign = 1 if side_i else -1
            body_position = np.array(links[legs_parents[leg_i]].pose[:3])
            # Shoulder 0
            pose = np.concatenate([
                body_position +  [
                    leg_offset[leg_i][0],
                    sign*leg_offset[leg_i][1],
                    leg_offset[leg_i][2]
                ],
                [0, 0, 0]
            ])
            index = convention.leglink2index(
                leg_i,
                side_i,
                0
            )
            links[index] = Link.sphere(
                name=convention.leglink2name(
                    leg_i,
                    side_i,
                    0
                ),
                radius=1.1*leg_radius,
                pose=pose,
                color=[0.7, 0.5, 0.5, 0.5]
            )
            links[index].inertial.mass = 0
            links[index].inertial.inertias = np.zeros(6)
            # Shoulder 1
            index = convention.leglink2index(
                leg_i,
                side_i,
                1
            )
            links[index] = Link.sphere(
                name=convention.leglink2name(
                    leg_i,
                    side_i,
                    1
                ),
                radius=1.3*leg_radius,
                pose=pose,
                color=[0.9, 0.9, 0.9, 0.3]
            )
            links[index].inertial.mass = 0
            links[index].inertial.inertias = np.zeros(6)
            # Shoulder 2
            shape_pose = [
                0, sign*(0.5*leg_length), 0,
                np.pi/2, 0, 0
            ]
            links[convention.leglink2index(
                leg_i,
                side_i,
                2
            )] = Link.capsule(
                name=convention.leglink2name(
                    leg_i,
                    side_i,
                    2
                ),
                length=leg_length,
                radius=leg_radius,
                pose=pose,
                # inertial_pose=shape_pose,
                shape_pose=shape_pose,
            )
            # Elbow
            pose = np.copy(pose)
            pose[1] += sign*leg_length
            links[convention.leglink2index(
                leg_i,
                side_i,
                3
            )] = Link.capsule(
                name=convention.leglink2name(
                    leg_i,
                    side_i,
                    3
                ),
                length=leg_length,
                radius=leg_radius,
                pose=pose,
                # inertial_pose=shape_pose,
                shape_pose=shape_pose,
                # color=[
                #     [[0.9, 0.0, 0.0, 1.0], [0.0, 0.9, 0.0, 1.0]],
                #     [[0.0, 0.0, 0.9, 1.0], [1.0, 0.7, 0.0, 1.0]]
                # ][leg_i][side_i]
            )

    # Leg joints
    for leg_i in range(options.morphology.n_legs//2):
        for side_i in range(2):
            sign = 1 if side_i else -1
            for joint_i in range(options.morphology.n_dof_legs):
                axis = [
                    [0, 0, sign],
                    [-sign, 0, 0],
                    [0, 1, 0],
                    [-sign, 0, 0]
                ]
                name = convention.legjoint2name(leg_i, side_i, joint_i)
                l_index = convention.leglink2index(leg_i, side_i, joint_i)
                j_index = convention.legjoint2index(leg_i, side_i, joint_i)
                p_index = (
                    legs_parents[leg_i]
                    if joint_i == 0
                    else convention.leglink2index(
                        leg_i,
                        side_i,
                        joint_i-1
                    )
                )
                joints[j_index] = Joint(
                    name=name,
                    joint_type='revolute',
                    parent=links[p_index],
                    child=links[l_index],
                    xyz=axis[joint_i],
                    limits=[
                        0 if joint_i == 3 else -0.5*np.pi,
                        0.5*np.pi,
                        max_torque,
                        max_velocity,
                    ],
                )

    # Use 2D
    constraint_links = [
        Link.empty(
            name='world',
            pose=[0, 0, 0, 0, 0, 0],
        ),
        Link.empty(
            name='world_2',
            pose=[0, 0, 0, 0, 0, 0],
        )
    ] if use_2d else []
    constraint_joints = [
        Joint(
            name='world_joint',
            joint_type='prismatic',
            parent=constraint_links[0],
            child=constraint_links[1],
            pose=[0, 0, 0, 0, 0, 0],
            xyz=[1, 0, 0],
            limits=np.array([-1, 1, 0, 1])
        ),
        Joint(
            name='world_joint2',
            joint_type='prismatic',
            parent=constraint_links[1],
            child=links[0],
            pose=[0, 0, 0, 0, 0, 0],
            xyz=[0, 0, 1],
            limits=np.array([-1, 1, 0, 1])
        )
    ] if use_2d else []

    # Create SDF
    sdf = ModelSDF(
        name=model_name,
        pose=np.concatenate([
            np.asarray([0, 0, 0.1])*scale,
            [0, 0, 0]
        ]),
        links=constraint_links+links,
        joints=constraint_joints+joints,
    )
    filename = create_new_model_from_farms_sdf(
        name=model_name,
        version=model_version,
        sdf=sdf,
        options={
            'author': 'Jonathan Arreguit',
            'email': 'jonathan.arreguitoneill@epfl.ch',
            'overwrite': True,
            'subfolders': ['amphibious'],
        }
    )
    print(filename)

    # Mass
    mass = 0
    for link in sdf.links:
        mass += link.inertial.mass
    print('Mass: {} [kg]'.format(mass))

    return filename


def generate_amphibious(model_name, model_version, **kwargs):
    """Generate amphibious"""

    # Arguments
    n_joints_body = kwargs.pop('n_joints_body')
    legs_parents = kwargs.pop('legs_parents')
    n_legs = 2*len(legs_parents)
    leg_offset = kwargs.pop('leg_offset', [0, 0.04, -0.02])
    leg_length = kwargs.pop('leg_length', 0.04)
    leg_radius = kwargs.pop('leg_radius', 0.01)

    # Animat options
    animat_options = AmphibiousOptions.from_options({
        'n_legs': n_legs,
        'n_dof_legs': 4,
        'n_joints_body': n_joints_body,
    })
    animat_options.morphology.mesh_directory = 'meshes_{}'.format(model_name)
    convention = AmphibiousConvention(**animat_options.morphology)
    # print(convention)
    # print(convention.n_joints_body)
    # print(convention.n_dof_legs)
    # print(convention.n_legs)
    # Generate SDF
    filepath = generate_sdf(
        model_name=model_name,
        model_version=model_version,
        leg_offset=leg_offset,
        leg_length=leg_length,
        leg_radius=leg_radius,
        legs_parents=legs_parents,
        options=animat_options,
        convention=convention,
        # scale=1,
        **kwargs
    )

    # Setup meshes
    original_meshes_path = os.path.join(
        get_model_path(name=model_name, version='body'),
        'sdf',
        'meshes_{}'.format(model_name),
    )
    meshes_output_path = os.path.join(
        os.path.dirname(filepath),
        animat_options.morphology.mesh_directory,
    )
    if os.path.isdir(meshes_output_path):
        shutil.rmtree(meshes_output_path, ignore_errors=True)
    shutil.copytree(
        original_meshes_path,
        meshes_output_path,
    )
