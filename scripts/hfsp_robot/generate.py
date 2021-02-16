"""Convert URDF to SDF"""

import os
from farms_sdf.sdf import (
    ModelSDF,
    Mesh,
    Visual,
    Inertial,
)
from farms_models.utils import (
    get_model_path,
    create_new_model_from_farms_sdf,
)


def main():
    """Main"""
    # Load URDF
    model_path = get_model_path(name='hfsp_robot', version='0')
    model = ModelSDF.from_urdf(os.path.join(
        model_path,
        'urdf',
        'SalamanderRobot.urdf',
    ))

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
