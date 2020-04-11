"""Arrays"""

cimport numpy as np
ctypedef double CTYPE
ctypedef double[:] CTYPEv1
ctypedef double[:, :] CTYPEv2
ctypedef double[:, :, :] CTYPEv3
ctypedef unsigned int INDEX
ctypedef unsigned int[:] INDEXv1
ctypedef unsigned int[:, :] INDEXv2
ctypedef unsigned int[:, :, :] INDEXv3
ctypedef np.float64_t DTYPE


cdef class NetworkArray:
    pass
    # cpdef public copy_array(self)
    # cpdef public log(self, times, folder, name, extension)


cdef class NetworkArray2D(NetworkArray):
    """Network array"""
    cdef readonly CTYPEv2 array


cdef class NetworkArray3D(NetworkArray):
    """Network array"""
    cdef readonly CTYPEv3 array


cdef class IntegerArray2D(NetworkArray):
    """Network array"""
    cdef readonly INDEXv2 array


cdef class IntegerArray3D(NetworkArray):
    """Network array"""
    cdef readonly INDEXv3 array
