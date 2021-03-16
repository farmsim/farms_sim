"""Convert URDF to SDF"""

import os
import shutil
from farms_sdf.sdf import ModelSDF
from farms_models.utils import (
    get_model_path,
    create_new_model_from_farms_sdf,
)


def generate(version):
    """Generate"""
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
    print('\nLinks:')
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

    # Polypterus version handling
    if 'polypterus' in version:

        # Girdle
        girdle_name = 'link_girdle_salamander_2_v31link_girdle_v71girdle_v31girdle1'
        girdle_mesh = 'meshes/link_girdle_Polypt_2_v21link_girdle_v71girdle_v31girdle1.stl'
        girdle_path = os.path.join(directory, girdle_mesh)
        for link in model.links:
            if link.name == girdle_name:
                visual = link.visuals[0]
                visual.geometry.uri = girdle_mesh
                link.collisions[0].geometry.uri = girdle_mesh
                link.inertial = inertial.from_mesh(
                    path=girdle_path,
                    scale=visual.geometry.scale[0],
                    pose=visual.pose,
                    units=inertial.units,
                    mass=0.15306822632853107,
                )
                break

        # Links from hindlimbs
        hindlinks = [
            'link_left_leg_v22XM430_W350_R_v1_v21X-430_IDLE1',
            'link_right_leg_v22XM430_W350_R_v1_v21X-430_IDLE1',
            'link_foot_v23fr12_h1011',
            'link_foot_v24fr12_h1011',
        ]

        # Links
        links_remove = []
        print('\nRemoving links:')
        for link in model.links:
            if link.name in hindlinks:
                print('  - {}'.format(link.name))
                links_remove.append(link)
        for link in links_remove:
            model.links.remove(link)

        # Joints
        joints_remove = []
        print('\nRemoving joints')
        for joint in model.joints:
            if joint.child in hindlinks:
                print('  - {}'.format(joint.name))
                joints_remove.append(joint)
        for joint in joints_remove:
            model.joints.remove(joint)

    # Write to SDF
    filename = create_new_model_from_farms_sdf(
        name='hfsp_robot',
        version=version,
        sdf=model,
        options={
            'author': 'Jonathan Arreguit',
            'email': 'jonathan.arreguitoneill@epfl.ch',
            'overwrite': True,
            'subfolders': ['amphibious'],
        }
    )
    print(filename)

    # Setup meshes
    original_meshes_path = os.path.join(model_path, 'urdf', 'meshes')
    meshes_output_path = os.path.join(
        os.path.dirname(filename),
        'meshes',
    )
    if os.path.isdir(meshes_output_path):
        shutil.rmtree(meshes_output_path, ignore_errors=True)
    shutil.copytree(original_meshes_path, meshes_output_path)

def main():
    """Main"""
    # generate(version='0')
    generate(version='salamander_0')
    generate(version='polypterus_0')


if __name__ == '__main__':
    main()
