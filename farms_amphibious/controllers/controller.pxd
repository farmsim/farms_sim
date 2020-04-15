"""Cython controller code"""

include "../data/types.pxd"
from ..data.animat_data_cy cimport (
    AnimatDataCy,
    NetworkParametersCy,
    OscillatorArrayCy,
    OscillatorConnectivityCy,
    ContactConnectivityCy,
    HydroConnectivityCy,
    ContactsArrayCy,
    HydrodynamicsArrayCy,
    JointsArrayCy,
)


cpdef void ode_dphase(
    CTYPEv1 state,
    CTYPEv1 dstate,
    OscillatorArrayCy oscillators,
    OscillatorConnectivityCy connectivity,
) nogil


cpdef void ode_damplitude(
    CTYPEv1 state,
    CTYPEv1 dstate,
    OscillatorArrayCy oscillators,
) nogil


cpdef void ode_contacts(
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
    CTYPEv1 state,
    CTYPEv1 dstate,
    JointsArrayCy joints,
    unsigned int n_oscillators,
) nogil


cpdef CTYPEv1 ode_oscillators_sparse(
    CTYPE time,
    CTYPEv1 state,
    unsigned int iteration,
    AnimatDataCy data,
    NetworkParametersCy network,
) nogil
