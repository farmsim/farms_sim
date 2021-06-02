"""Convert URDF to SDF"""

import os
from farms_sdf.sdf import ModelSDF, Mesh, Visual, Inertial
from farms_models.utils import get_model_path, create_new_model_from_farms_sdf
from farms_amphibious.utils.parse_args import parse_args_model_gen


def main():
    """Main"""
    clargs = parse_args_model_gen(description='Generate Orobot')

    # Load URDF
    model_path = (
        clargs.model_path
        if clargs.model_path
        else get_model_path(name='orobot', version='0')
    )
    urdf_original = (
        clargs.original
        if clargs.original
        else os.path.join(model_path, 'urdf', 'orobot_final.urdf')
    )
    assert os.path.isfile(urdf_original), urdf_original
    model = ModelSDF.from_urdf(urdf_original)

    # Final corrections
    print('\nLinks:')
    for link in model.links:
        for child in link.visuals + link.collisions:
            if isinstance(child.geometry, Mesh):
                child.geometry.uri = child.geometry.uri.replace(
                    'package://krock_urdf/orobot_script_meshes/',
                    '../meshes/',
                )
        print('\'{}\','.format(link.name))

    # Add eyes
    for link in model.links:
        if link.name == 'HEAD_TT_C':
            eye = Visual.sphere(
                'eye_left',
                pose=[0.1, 0.34, -0.05, 0, 0, 0],
                units=link.units,
                radius=0.02,
                color=[0, 0, 0, 1],
            )
            link.visuals.append(eye)
            eye = Visual.sphere(
                'eye_right',
                pose=[0.1, 0.34, 0.05, 0, 0, 0],
                units=link.units,
                radius=0.015,
                color=[0, 0, 0, 1],
            )
            link.visuals.append(eye)

        if not link.inertial:
            link.inertial = Inertial.empty()
            if link.name == 'base_link':
                link.inertial.mass = 1e-3

    print('\nJoints:')
    for joint in model.joints:
        if joint.type != 'fixed':
            print('\'{}\','.format(joint.name))

    # Write to SDF
    # model.write(filename='../sdf/orobot.sdf')
    filename = create_new_model_from_farms_sdf(
        name='orobot',
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
    main()
