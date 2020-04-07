"""Cython controller code"""

cimport numpy as np

from ..data.animat_data_cy cimport (
    AnimatDataCy,
    NetworkParametersCy,
    OscillatorArrayCy,
    ConnectivityArrayCy,
    ContactsArrayCy,
    HydrodynamicsArrayCy,
    JointsArrayCy,
)


ctypedef double CTYPE
ctypedef np.float64_t DTYPE


cpdef void ode_dphase(
    CTYPE[:] state,
    double[:] dstate,
    OscillatorArrayCy oscillators,
    ConnectivityArrayCy connectivity,
    unsigned int n_oscillators,
) nogil


cpdef void ode_damplitude(
    CTYPE[:] state,
    double[:] dstate,
    OscillatorArrayCy oscillators,
    unsigned int n_oscillators,
) nogil


cpdef void ode_contacts(
    unsigned int iteration,
    CTYPE[:] state,
    double[:] dstate,
    ContactsArrayCy contacts,
    ConnectivityArrayCy contacts_connectivity,
) nogil


cpdef void ode_hydro(
    unsigned int iteration,
    CTYPE[:] state,
    double[:] dstate,
    HydrodynamicsArrayCy hydrodynamics,
    ConnectivityArrayCy hydro_connectivity,
    unsigned int n_oscillators,
) nogil


cpdef void ode_joints(
    CTYPE[:] state,
    double[:] dstate,
    JointsArrayCy joints,
    unsigned int n_oscillators,
) nogil


cpdef double[:] ode_oscillators_sparse(
    double time,
    CTYPE[:] state,
    AnimatDataCy data,
    NetworkParametersCy network,
)
