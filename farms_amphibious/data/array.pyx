"""Arrays"""

import os
import numpy as np
cimport numpy as np


cdef class NetworkArray(object):
    """Network array"""

    cpdef unsigned int size(self, unsigned int index):
        """Shape"""
        return self.array.shape[index]

    def copy_array(self):
        """Copy array"""
        return np.copy(self.array)

    def log(self, times, folder, name, extension):
        """Log data"""
        os.makedirs(folder, exist_ok=True)
        if extension == "npy":
            save_function = np.save
            nosplit = True
        elif extension in ("txt", "csv"):
            save_function = np.savetxt
            nosplit = False
        else:
            raise Exception(
                "Format {} is not valid for logging array".format(extension)
            )
        if nosplit or self.array.ndim == 2:
            path = folder + "/" + name + "." + extension
            save_function(path, self.array[:len(times)])
        elif self.array.ndim == 3:
            for i in range(np.shape(self.array)[1]):
                path = folder+"/"+name+"_{}.".format(i)+extension
                save_function(path, self.array[:len(times), i])
        else:
            msg = "Dimensionality {} is not valid for extension of type {}"
            raise Exception(msg.format(self.array.ndim, extension))


cdef class NetworkArray1D(NetworkArray):
    """Network array"""

    def __init__(self, array):
        super(NetworkArray1D, self).__init__()
        self.array = array


cdef class NetworkArray2D(NetworkArray):
    """Network array"""

    def __init__(self, array):
        super(NetworkArray2D, self).__init__()
        self.array = array


cdef class NetworkArray3D(NetworkArray):
    """Network array"""

    def __init__(self, array):
        super(NetworkArray3D, self).__init__()
        self.array = array


cdef class IntegerArray2D(NetworkArray):
    """Network array"""

    def __init__(self, array):
        super(IntegerArray2D, self).__init__()
        self.array = array
