"""Cython controller code"""

include "../data/array.pxd"
from ..data.animat_data_cy cimport (
    AnimatDataCy,
    NetworkParametersCy,
    OscillatorArrayCy,
    ConnectivityArrayCy,
    ContactsArrayCy,
    HydrodynamicsArrayCy,
    JointsArrayCy,
)


cpdef void ode_dphase(
    CTYPEv1 state,
    CTYPEv1 dstate,
    OscillatorArrayCy oscillators,
    ConnectivityArrayCy connectivity,
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
    ConnectivityArrayCy contacts_connectivity,
) nogil


cpdef void ode_hydro(
    unsigned int iteration,
    CTYPEv1 state,
    CTYPEv1 dstate,
    HydrodynamicsArrayCy hydrodynamics,
    ConnectivityArrayCy hydro_connectivity,
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
    AnimatDataCy data,
    NetworkParametersCy network,
) nogil


cpdef void rk_oscillators(
    CTYPE time,
    CTYPE timestep,
    CTYPEv1 state,
    AnimatDataCy data,
    NetworkParametersCy network,
    CTYPEv1 k1,
    CTYPEv1 k2,
    CTYPEv1 k3,
    CTYPEv1 k4,
    CTYPEv1 state_out,
) nogil
