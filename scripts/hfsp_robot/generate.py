"""Convert URDF to SDF"""

import os
from farms_sdf.sdf import (
    ModelSDF,
    # Mesh,
    # Visual,
    # Inertial,
)
from farms_models.utils import (
    get_model_path,
    create_new_model_from_farms_sdf,
)


def main():
    """Main"""
    # Load URDF
    model_path = get_model_path(name='hfsp_robot', version='0')
    urdf_parth = os.path.join(
        model_path,
        'urdf',
        'V2SalamanderRobot.urdf'
        # 'SalamanderRobot.urdf',
    )
    model = ModelSDF.from_urdf(urdf_parth)
    directory = os.path.dirname(urdf_parth)

    # Links
    print('Links:')
    for link in model.links:
        print("'{}',".format(link.name))
        if link.visuals:
            assert len(link.visuals) == 1
            visual = link.visuals[0]
            inertial = link.inertial
            mesh_path = os.path.join(directory, visual.geometry.uri)
            link.inertial = inertial.from_mesh(
                path=mesh_path,
                scale=visual.geometry.scale[0],
                pose=visual.pose,
                units=inertial.units,
                mass=inertial.mass,
            )
            # link.inertial.inertias = inertial.inertias
        else:
            link.inertial.inertias = [0, 0, 0, 0, 0, 0]

    # Joints
    print('\nJoints:')
    for joint in model.joints:
        print("'{}',".format(joint.name))

    # Write to SDF
    filename = create_new_model_from_farms_sdf(
        name='hfsp_robot',
        version='0',
        sdf=model,
        options={
            'author': 'Jonathan Arreguit',
            'email': 'jonathan.arreguitoneill@epfl.ch',
            'overwrite': True,
        }
    )
    print(filename)


if __name__ == '__main__':
    main()
