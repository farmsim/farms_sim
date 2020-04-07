"""Arrays"""

cimport numpy as np
ctypedef double CTYPE
ctypedef double[:] CTYPEv1
ctypedef double[:, :] CTYPEv2
ctypedef double[:, :, :] CTYPEv3
ctypedef np.float64_t DTYPE


cdef class NetworkArray:
    pass


cdef class NetworkArray2D(NetworkArray):
    """Network array"""
    cdef readonly CTYPEv2 array
    cdef readonly unsigned int[2] size


cdef class NetworkArray3D(NetworkArray):
    """Network array"""
    cdef readonly CTYPEv3 array
    cdef readonly unsigned int[3] size
