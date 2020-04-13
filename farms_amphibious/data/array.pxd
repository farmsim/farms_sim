"""Arrays"""

include "types.pxd"


cdef class NetworkArray:
    """Network array"""
    cpdef public unsigned int size(self, unsigned int index)


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
