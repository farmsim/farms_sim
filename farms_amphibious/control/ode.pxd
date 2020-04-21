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
    DTYPEv1 state,
    DTYPEv1 dstate,
    DriveArrayCy drives,
    OscillatorsCy oscillators,
    OscillatorConnectivityCy connectivity,
) nogil


cpdef void ode_damplitude(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    DriveArrayCy drives,
    OscillatorsCy oscillators,
) nogil


cpdef void ode_contacts(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    ContactsArrayCy contacts,
    ContactConnectivityCy contacts_connectivity,
) nogil


cpdef void ode_contacts_tegotae(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    ContactsArrayCy contacts,
    ContactConnectivityCy contacts_connectivity,
) nogil


cpdef void ode_hydro(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    HydrodynamicsArrayCy hydrodynamics,
    HydroConnectivityCy hydro_connectivity,
    unsigned int n_oscillators,
) nogil


cpdef void ode_joints(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    DriveArrayCy drives,
    JointsArrayCy joints,
    unsigned int n_oscillators,
) nogil


cpdef DTYPEv1 ode_oscillators_sparse(
    DTYPE time,
    DTYPEv1 state,
    DTYPEv1 dstate,
    unsigned int iteration,
    AnimatDataCy data,
) nogil


cpdef DTYPEv1 ode_oscillators_sparse_no_sensors(
    DTYPE time,
    DTYPEv1 state,
    DTYPEv1 dstate,
    unsigned int iteration,
    AnimatDataCy data,
) nogil


cpdef DTYPEv1 ode_oscillators_sparse_tegotae(
    DTYPE time,
    DTYPEv1 state,
    DTYPEv1 dstate,
    unsigned int iteration,
    AnimatDataCy data,
) nogil
