"""Cython code"""

import time
# import numpy as np

cimport cython
cimport numpy as np

from libc.math cimport sin, fabs  # cos,
# from libc.stdlib cimport malloc, free
# from cython.parallel import prange


cpdef void ode_dphase(
    CTYPE[:] state,
    double[:] dstate,
    OscillatorArrayCy oscillators,
    ConnectivityArrayCy connectivity,
    unsigned int n_oscillators,
) nogil:
    """Oscillator phase ODE

    d_theta = omega + sum amplitude_j*weight*sin(phase_j - phase_i - phase_bias)

    """
    cdef unsigned int i, i0, i1
    for i in range(n_oscillators):  # , nogil=True):
        # Intrinsic frequency
        dstate[i] = oscillators.array[0][i]
    for i in range(connectivity.size[0]):
        i0 = <unsigned int> (connectivity.array[i][0] + 0.5)
        i1 = <unsigned int> (connectivity.array[i][1] + 0.5)
        # amplitude_j*weight*sin(phase_j - phase_i - phase_bias)
        dstate[i0] += state[n_oscillators+i1]*connectivity.array[i][2]*sin(
            state[i1] - state[i0]
            - connectivity.array[i][3]
        )


cpdef void ode_damplitude(
    CTYPE[:] state,
    double[:] dstate,
    OscillatorArrayCy oscillators,
    unsigned int n_oscillators,
) nogil:
    """Oscillator amplitude ODE

    d_amplitude = rate*(nominal_amplitude - amplitude)

    """
    cdef unsigned int i
    for i in range(n_oscillators):  # , nogil=True):
        # rate*(nominal_amplitude - amplitude)
        dstate[n_oscillators+i] = oscillators.array[1][i]*(
            oscillators.array[2][i] - state[n_oscillators+i]
        )


cpdef void ode_contacts(
    unsigned int iteration,
    CTYPE[:] state,
    double[:] dstate,
    ContactsArrayCy contacts,
    ConnectivityArrayCy contacts_connectivity,
) nogil:
    """Sensory feedback - Contacts

    Can affect d_phase and d_amplitude

    """
    cdef double contact
    cdef unsigned int i, i0, i1
    for i in range(contacts_connectivity.size[0]):
        i0 = <unsigned int> (
            contacts_connectivity.array[i][0] + 0.5
        )
        i1 = <unsigned int> (
            contacts_connectivity.array[i][1] + 0.5
        )
        # contact_weight*contact_force
        # contact = (
        #     contacts.array[iteration][i1][0]**2
        #     + contacts.array[iteration][i1][1]**2
        #     + contacts.array[iteration][i1][2]**2
        # )**0.5
        contact = fabs(contacts.array[iteration][i1][2])
        dstate[i0] += (
            contacts_connectivity.array[i][2]
            *(10*contact/(1+10*contact))  # Saturation
            # *cos(state[i0])
            # *sin(state[i0])  # For Tegotae
        )


cpdef void ode_hydro(
    unsigned int iteration,
    CTYPE[:] state,
    double[:] dstate,
    HydrodynamicsArrayCy hydrodynamics,
    ConnectivityArrayCy hydro_connectivity,
    unsigned int n_oscillators,
) nogil:
    """Sensory feedback - Hydrodynamics

    Can affect d_phase and d_amplitude

    """
    cdef double hydro_force
    cdef unsigned int i, i0, i1
    for i in range(hydro_connectivity.size[0]):
        i0 = <unsigned int> (hydro_connectivity.array[i][0] + 0.5)
        i1 = <unsigned int> (hydro_connectivity.array[i][1] + 0.5)
        hydro_force = fabs(hydrodynamics.array[iteration][i1][1])
        # dfrequency += hydro_weight*hydro_force
        dstate[i0] += hydro_connectivity.array[i][2]*hydro_force
        # damplitude += hydro_weight*hydro_force
        dstate[n_oscillators+i0] += hydro_connectivity.array[i][3]*hydro_force


cpdef void ode_joints(
    CTYPE[:] state,
    double[:] dstate,
    JointsArrayCy joints,
    unsigned int n_oscillators,
) nogil:
    """Joints offset

    d_joints_offset = rate*(joints_offset_desired - joints_offset)

    """
    cdef unsigned int i
    for i in range(joints.size[1]):
        # rate*(joints_offset_desired - joints_offset)
        dstate[2*n_oscillators+i] = joints.array[1][i]*(
            joints.array[0][i] - state[2*n_oscillators+i]
        )


cpdef double[:] ode_oscillators_sparse(
    double time,
    CTYPE[:] state,
    AnimatDataCy data,
    NetworkParametersCy network,
):
    """ODE"""
    cdef unsigned int n_oscillators = data.network.oscillators.size[1]
    cdef double[:] dstate = data.state.array[data.iteration][1]
    ode_dphase(
        state,
        dstate,
        data.network.oscillators,
        data.network.connectivity,
        n_oscillators,
    )
    ode_damplitude(
        state,
        dstate,
        data.network.oscillators,
        n_oscillators,
    )
    ode_contacts(
        data.iteration,
        state,
        dstate,
        data.sensors.contacts,
        data.network.contacts_connectivity,
    )
    ode_hydro(
        data.iteration,
        state,
        dstate,
        data.sensors.hydrodynamics,
        data.network.hydro_connectivity,
        n_oscillators,
    )
    ode_joints(
        state,
        dstate,
        data.joints,
        n_oscillators,
    )
    return dstate
