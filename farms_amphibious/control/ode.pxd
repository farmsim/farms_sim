"""Cython controller code"""

include '../data/types.pxd'
from ..data.animat_data_cy cimport (
    AnimatDataCy,
    NetworkParametersCy,
    DriveArrayCy,
    OscillatorsCy,
    OscillatorConnectivityCy,
    ContactConnectivityCy,
    HydroConnectivityCy,
    ContactsArrayCy,
    HydrodynamicsArrayCy,
    JointsArrayCy,
)


cpdef void ode_dphase(
    unsigned int iteration,
    CTYPEv1 state,
    CTYPEv1 dstate,
    DriveArrayCy drives,
    OscillatorsCy oscillators,
    OscillatorConnectivityCy connectivity,
) nogil


cpdef void ode_damplitude(
    unsigned int iteration,
    CTYPEv1 state,
    CTYPEv1 dstate,
    DriveArrayCy drives,
    OscillatorsCy oscillators,
) nogil


cpdef void ode_contacts(
    unsigned int iteration,
    CTYPEv1 state,
    CTYPEv1 dstate,
    ContactsArrayCy contacts,
    ContactConnectivityCy contacts_connectivity,
) nogil


cpdef void ode_contacts_tegotae(
    unsigned int iteration,
    CTYPEv1 state,
    CTYPEv1 dstate,
    ContactsArrayCy contacts,
    ContactConnectivityCy contacts_connectivity,
) nogil


cpdef void ode_hydro(
    unsigned int iteration,
    CTYPEv1 state,
    CTYPEv1 dstate,
    HydrodynamicsArrayCy hydrodynamics,
    HydroConnectivityCy hydro_connectivity,
    unsigned int n_oscillators,
) nogil


cpdef void ode_joints(
    unsigned int iteration,
    CTYPEv1 state,
    CTYPEv1 dstate,
    DriveArrayCy drives,
    JointsArrayCy joints,
    unsigned int n_oscillators,
) nogil


cpdef CTYPEv1 ode_oscillators_sparse(
    CTYPE time,
    CTYPEv1 state,
    CTYPEv1 dstate,
    unsigned int iteration,
    AnimatDataCy data,
) nogil


cpdef CTYPEv1 ode_oscillators_sparse_no_sensors(
    CTYPE time,
    CTYPEv1 state,
    CTYPEv1 dstate,
    unsigned int iteration,
    AnimatDataCy data,
) nogil


cpdef CTYPEv1 ode_oscillators_sparse_tegotae(
    CTYPE time,
    CTYPEv1 state,
    CTYPEv1 dstate,
    unsigned int iteration,
    AnimatDataCy data,
) nogil
