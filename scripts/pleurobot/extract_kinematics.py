"""Extract kienmatics"""

import os
import numpy as np
import matplotlib.pyplot as plt

from farms_models.utils import get_model_path


def main(show=True):
    """Main"""
    directory = get_model_path('pleurobot', '1')
    kin_directory = os.path.join(directory, 'kinematics')
    data = np.loadtxt(os.path.join(kin_directory, 'spine.txt'))
    data = np.insert(arr=data, obj=12, values=0, axis=1)
    data = np.insert(arr=data, obj=6, values=0, axis=1)
    data[:, 15] = -data[:, 15] + np.pi/2
    data[:, 19] = data[:, 19] - np.pi/2
    data[:, 24] = data[:, 24] + np.pi/2
    data[:, 28] = -data[:, 28] - np.pi/2
    data = np.radians(data)
    data = np.concatenate([data]*10, axis=0)
    np.savetxt(os.path.join(kin_directory, 'kinematics.csv'), data)
    if show:
        plt.plot(data)
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Joint angle [rad]')
        plt.show()


if __name__ == '__main__':
    main()
