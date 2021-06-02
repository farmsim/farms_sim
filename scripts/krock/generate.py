"""Convert URDF to SDF"""

import os
import numpy as np
from farms_sdf.sdf import ModelSDF, Mesh, Inertial
from farms_models.utils import get_model_path, create_new_model_from_farms_sdf
from farms_amphibious.utils.parse_args import parse_args_model_gen


def convert_from_urdf_old():
    """Main"""
    clargs = parse_args_model_gen(description='Generate Korck')

    # Load URDF
    model_path = (
        clargs.model_path
        if clargs.model_path
        else get_model_path(name='krock', version='0')
    )
    model = ModelSDF.from_urdf(os.path.join(
        model_path, 'urdf', 'krock_final.urdf'
    ))
    # print(model)

    # Correct links positions
    pose = None
    for link in model.links:
        if link.name == 'krock2':
            pose = np.array(link.collisions[0].pose[:3])
            break
    assert pose is not None, 'Pose not found'
    for link in model.links:
        if np.allclose(link.pose[:3], [0, 0, 0]):
            link.pose[:3] = pose
            for visual in link.visuals:
                visual.pose = (np.array(visual.pose[:3]) - pose).tolist() + visual.pose[3:]
            for collision in link.collisions:
                collision.pose = (np.array(collision.pose[:3]) - pose).tolist() + collision.pose[3:]

    # Correct tail
    found = False
    for link in model.links:
        if link.name == 'Fin_Transform':
            link.pose[0] -= 0.15
            found = True
            break
    assert found, 'Link Fin_Transform was not found'
    found = False
    for joint in model.joints:
        if joint.child == 'Fin_Transform':
            joint.parent = 'solid_tail3_endpoint'
            found = True
            break
    assert found, 'Joint for Fin_Transform was not found'

    # Correct inertials
    mass = 0
    print('\nLinks with inertials:')
    for link in model.links:
        for child in link.visuals + link.collisions:
            if isinstance(child.geometry, Mesh):
                child.geometry.uri = child.geometry.uri.replace(
                    'package://krock_urdf/krock_script_meshes/',
                    '../meshes/',
                )

        if link.inertial:
            if link.name == 'HRknee_HJ_C':
                link.inertial.mass = 0.1
            mass += link.inertial.mass
            if link.visuals:
                visual = link.visuals[0]
                inertial = link.inertial
                mesh_path = visual.geometry.uri
                link.inertial = inertial.from_mesh(
                    path=os.path.join(model_path, 'meshes', mesh_path),
                    scale=visual.geometry.scale[0],
                    pose=visual.pose,
                    units=inertial.units,
                    mass=inertial.mass,
                )
                # link.inertial.inertias = inertial.inertias
            elif link.collisions:
                collision = link.collisions[0]
                inertial = link.inertial
                link.inertial = inertial.box(
                    size=collision.geometry.size,
                    pose=collision.pose,
                    units=inertial.units,
                    mass=inertial.mass,
                )
            else:
                raise Exception('No visual or collision found')
            print('\'{}\','.format(link.name))
        elif link.name != 'base_link':
            link.inertial = Inertial.empty(link.units)
    print('\nTotal mass: {} [kg]'.format(mass))

    print('\nJoints:')
    for joint in model.joints:
        if joint.type != 'fixed':
            print('\'{}\','.format(joint.name))

    # Write to SDF
    # model.write(filename='../sdf/krock.sdf')
    filename = create_new_model_from_farms_sdf(
        name='krock',
        version='0',
        sdf=model,
        sdf_path=clargs.sdf_path,
        model_path=clargs.model_path,
        options={
            'author': 'Jonathan Arreguit',
            'email': 'jonathan.arreguitoneill@epfl.ch',
            'overwrite': True,
        }
    )
    print(filename)


def convert_from_sdf():
    """Main"""
    clargs = parse_args_model_gen(description='Generate Korck')

    # Load URDF
    model_path = (
        clargs.model_path
        if clargs.model_path
        else get_model_path(name='krock', version='0')
    )
    sdf_original = (
        clargs.original
        if clargs.original
        else os.path.join(model_path, 'design', 'sdf', 'krock.sdf')
    )
    assert os.path.isfile(sdf_original), sdf_original
    model = ModelSDF.read(sdf_original)[0]

    # Write to SDF
    filename = create_new_model_from_farms_sdf(
        name='krock',
        version='0',
        sdf=model,
        sdf_path=clargs.sdf_path,
        model_path=clargs.model_path,
        options={
            'author': 'Jonathan Arreguit',
            'email': 'jonathan.arreguitoneill@epfl.ch',
            'overwrite': True,
        }
    )
    print(filename)


if __name__ == '__main__':
    convert_from_sdf()
