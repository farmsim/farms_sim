"""Animat data"""

include "array.pxd"
from .array cimport NetworkArray2D, NetworkArray3D


cdef class AnimatDataCy:
    """Network parameter"""
    cdef public OscillatorNetworkStateCy state
    cdef public NetworkParametersCy network
    cdef public JointsArrayCy joints
    cdef public SensorsDataCy sensors
    cdef public unsigned int iteration


cdef class NetworkParametersCy:
    """Network parameter"""
    cdef public OscillatorArrayCy oscillators
    cdef public ConnectivityArrayCy connectivity
    cdef public ConnectivityArrayCy contacts_connectivity
    cdef public ConnectivityArrayCy hydro_connectivity


cdef class OscillatorNetworkStateCy(NetworkArray3D):
    """Network state"""
    cdef public unsigned int n_oscillators
    cdef public unsigned int _iterations


cdef class OscillatorArrayCy(NetworkArray2D):
    """Oscillator array"""
    pass


cdef class ConnectivityArrayCy(NetworkArray2D):
    """Connectivity array"""
    pass


cdef class JointsArrayCy(NetworkArray2D):
    """Oscillator array"""
    pass


cdef class SensorsDataCy:
    """SensorsData"""
    cdef public ContactsArrayCy contacts
    cdef public ProprioceptionArrayCy proprioception
    cdef public GpsArrayCy gps
    cdef public HydrodynamicsArrayCy hydrodynamics


cdef class ContactsArrayCy(NetworkArray3D):
    """Sensor array"""

    cpdef CTYPEv1 reaction(self, unsigned int iteration, unsigned int sensor_i)
    cpdef CTYPEv2 reaction_all(self, unsigned int sensor_i)
    cpdef CTYPEv1 friction(self, unsigned int iteration, unsigned int sensor_i)
    cpdef CTYPEv2 friction_all(self, unsigned int sensor_i)
    cpdef CTYPEv1 total(self, unsigned int iteration, unsigned int sensor_i)
    cpdef CTYPEv2 total_all(self, unsigned int sensor_i)


cdef class ProprioceptionArrayCy(NetworkArray3D):
    """Proprioception array"""

    cpdef CTYPE position(self, unsigned int iteration, unsigned int joint_i)
    cpdef CTYPEv1 positions(self, unsigned int iteration)
    cpdef CTYPEv2 positions_all(self)
    cpdef CTYPE velocity(self, unsigned int iteration, unsigned int joint_i)
    cpdef CTYPEv1 velocities(self, unsigned int iteration)
    cpdef CTYPEv2 velocities_all(self)
    cpdef CTYPEv1 force(self, unsigned int iteration, unsigned int joint_i)
    cpdef CTYPEv3 forces_all(self)
    cpdef CTYPEv1 torque(self, unsigned int iteration, unsigned int joint_i)
    cpdef CTYPEv3 torques_all(self)
    cpdef CTYPE motor_torque(self, unsigned int iteration, unsigned int joint_i)
    cpdef CTYPEv2 motor_torques(self)


cdef class GpsArrayCy(NetworkArray3D):
    """Gps array"""

    cpdef public CTYPEv1 com_position(self, unsigned int iteration, unsigned int link_i)
    cpdef public CTYPEv1 com_orientation(self, unsigned int iteration, unsigned int link_i)
    cpdef public CTYPEv1 urdf_position(self, unsigned int iteration, unsigned int link_i)
    cpdef public CTYPEv3 urdf_positions(self)
    cpdef public CTYPEv1 urdf_orientation(self, unsigned int iteration, unsigned int link_i)
    cpdef public CTYPEv1 com_lin_velocity(self, unsigned int iteration, unsigned int link_i)
    cpdef public CTYPEv3 com_lin_velocities(self)
    cpdef public CTYPEv1 com_ang_velocity(self, unsigned int iteration, unsigned int link_i)


cdef class HydrodynamicsArrayCy(NetworkArray3D):
    """Hydrodynamics array"""

    cpdef public CTYPEv3 forces(self)
    cpdef public CTYPEv3 torques(self)
