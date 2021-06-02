"""Correct inertias"""

import os
import numpy as np
from farms_sdf.sdf import ModelSDF, Visual
from farms_models.utils import get_sdf_path, create_new_model_from_farms_sdf
from farms_amphibious.utils.parse_args import parse_args_model_gen


def main():
    """Main"""
    clargs = parse_args_model_gen(description='Generate Pleurobot')
    filename = (
        clargs.original
        if clargs.original
        else get_sdf_path(name='pleurobot', version='0')
    )
    sdf = ModelSDF.read(filename=filename)[0]
    directory = os.path.dirname(filename)

    # Correct inertias
    for link in sdf.links:
        if link.visuals:
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

    # Add eyes
    for link in sdf.links:
        if link.name == 'Head':
            eye = Visual.sphere(
                'eye_left',
                pose=[0.16, -0.07, 0, 0, 0, 0],
                units=link.units,
                radius=0.05,
                color=[0, 0, 0, 1],
            )
            link.visuals.append(eye)
            eye = Visual.sphere(
                'eye_right',
                pose=[0.16, 0.07, 0, 0, 0, 0],
                units=link.units,
                radius=0.05,
                color=[0, 0, 0, 1],
            )
            link.visuals.append(eye)
            break

    # Write
    filename = create_new_model_from_farms_sdf(
        name='pleurobot',
        version='1',
        sdf=sdf,
        sdf_path=clargs.sdf_path,
        model_path=clargs.model_path,
        options={
            'author': 'Jonathan Arreguit',
            'email': 'jonathan.arreguitoneill.@epfl.ch',
            'overwrite': True,
        }
    )
    print(filename)


if __name__ == '__main__':
    main()
