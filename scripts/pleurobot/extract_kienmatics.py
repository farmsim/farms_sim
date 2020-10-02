"""Extract kienmatics"""

import os
import numpy as np
import matplotlib.pyplot as plt

from farms_models.utils import get_model_path


def main(show=True):
    """Main"""
    directory = get_model_path('pleurobot', '1')
    kin_directory = os.path.join(directory, 'kinematics')
    data, data_lf, data_rf, data_lh, data_rh = [
        np.loadtxt(os.path.join(
            kin_directory,
            '{}thetas{}.txt'.format('s' if suffix else '', suffix),
        ))
        for suffix in ['', 'LF', 'RF', 'LH', 'RH']
    ]
    # data_legs = np.concatenate([
    #     data_lf,
    #     data_rf,
    #     data_lh,
    #     data_rh
    # ], axis=1)
    data = np.insert(arr=data, obj=12, values=0, axis=1)
    data = np.insert(arr=data, obj=6, values=0, axis=1)
    data = 0.5*np.radians(data)
    data = np.concatenate([data]*10, axis=0)
    np.savetxt(os.path.join(kin_directory, 'kinematics.csv'), data)
    if show:
        plt.plot(data)
        plt.show()


if __name__ == '__main__':
    main()
