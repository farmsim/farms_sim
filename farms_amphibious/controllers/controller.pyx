"""Cython code"""

import time
# import numpy as np

cimport cython
cimport numpy as np

from libc.math cimport sin, fabs  # cos,
# from libc.stdlib cimport malloc, free
# from cython.parallel import prange


cdef inline void ode_dphase(
    double time,
    CTYPE[:] state,
    double[:] dstate,
    AnimatDataCy data,
    unsigned int n_oscillators,
) nogil:
    """Oscillator phase ODE

    d_theta = omega + sum amplitude_j*weight*sin(phase_j - phase_i - phase_bias)

    """
    cdef unsigned int i, i0, i1
    for i in range(n_oscillators):  # , nogil=True):
        # Intrinsic frequency
        dstate[i] = data.network.oscillators.array[0][i]
    for i in range(data.network.connectivity.size[0]):
        i0 = <unsigned int> (data.network.connectivity.array[i][0] + 0.5)
        i1 = <unsigned int> (data.network.connectivity.array[i][1] + 0.5)
        # amplitude_j*weight*sin(phase_j - phase_i - phase_bias)
        dstate[i0] += state[n_oscillators+i1]*data.network.connectivity.array[i][2]*sin(
            state[i1] - state[i0]
            - data.network.connectivity.array[i][3]
        )


cdef inline void ode_damplitude(
    double time,
    CTYPE[:] state,
    double[:] dstate,
    AnimatDataCy data,
    unsigned int n_oscillators,
) nogil:
    """Oscillator amplitude ODE

    d_amplitude = rate*(nominal_amplitude - amplitude)

    """
    cdef unsigned int i
    for i in range(n_oscillators):  # , nogil=True):
        # rate*(nominal_amplitude - amplitude)
        dstate[n_oscillators+i] = data.network.oscillators.array[1][i]*(
            data.network.oscillators.array[2][i] - state[n_oscillators+i]
        )


cdef inline void ode_contacts(
    double time,
    CTYPE[:] state,
    double[:] dstate,
    AnimatDataCy data,
    unsigned int n_oscillators,
) nogil:
    """Sensory feedback - Contacts

    Can affect d_phase and d_amplitude

    """
    cdef double contact
    cdef unsigned int i, i0, i1
    for i in range(data.network.contacts_connectivity.size[0]):
        i0 = <unsigned int> (
            data.network.contacts_connectivity.array[i][0] + 0.5
        )
        i1 = <unsigned int> (
            data.network.contacts_connectivity.array[i][1] + 0.5
        )
        # contact_weight*contact_force
        # contact = (
        #     data.sensors.contacts.array[data.iteration][i1][0]**2
        #     + data.sensors.contacts.array[data.iteration][i1][1]**2
        #     + data.sensors.contacts.array[data.iteration][i1][2]**2
        # )**0.5
        contact = fabs(data.sensors.contacts.array[data.iteration][i1][2])
        dstate[i0] += (
            data.network.contacts_connectivity.array[i][2]
            *(10*contact/(1+10*contact))  # Saturation
            # *cos(state[i0])
            # *sin(state[i0])  # For Tegotae
        )


cdef inline void ode_hydro(
    double time,
    CTYPE[:] state,
    double[:] dstate,
    AnimatDataCy data,
    unsigned int n_oscillators,
) nogil:
    """Sensory feedback - Hydrodynamics

    Can affect d_phase and d_amplitude

    """
    cdef double hydro_force
    cdef unsigned int i, i0, i1
    for i in range(data.network.hydro_connectivity.size[0]):
        i0 = <unsigned int> (
            data.network.hydro_connectivity.array[i][0] + 0.5
        )
        i1 = <unsigned int> (
            data.network.hydro_connectivity.array[i][1] + 0.5
        )
        hydro_force = fabs(
            data.sensors.hydrodynamics.array[data.iteration][i1][1]
        )
        # dfrequency += hydro_weight*hydro_force
        dstate[i0] += data.network.hydro_connectivity.array[i][2]*hydro_force
        # damplitude += hydro_weight*hydro_force
        dstate[n_oscillators+i0] += (
            data.network.hydro_connectivity.array[i][3]*hydro_force
        )


cdef inline void ode_joints(
    double time,
    CTYPE[:] state,
    double[:] dstate,
    AnimatDataCy data,
    unsigned int n_oscillators,
) nogil:
    """Joints offset

    d_joints_offset = rate*(joints_offset_desired - joints_offset)

    """
    cdef unsigned int i
    for i in range(data.joints.size[1]):
        # rate*(joints_offset_desired - joints_offset)
        dstate[2*n_oscillators+i] = data.joints.array[1][i]*(
            data.joints.array[0][i] - state[2*n_oscillators+i]
        )


cpdef double[:] ode_oscillators_sparse(
    double time,
    CTYPE[:] state,
    AnimatDataCy data,
) nogil:
    """ODE"""
    cdef unsigned int i, i0, i1
    cdef unsigned int n_oscillators = data.network.oscillators.size[1]
    cdef double[:] dstate = data.state.array[data.iteration][1]
    ode_dphase(
        time,
        state,
        dstate,
        data,
        n_oscillators,
    )
    ode_damplitude(
        time,
        state,
        dstate,
        data,
        n_oscillators,
    )
    ode_contacts(
        time,
        state,
        dstate,
        data,
        n_oscillators,
    )
    ode_hydro(
        time,
        state,
        dstate,
        data,
        n_oscillators,
    )
    ode_joints(
        time,
        state,
        dstate,
        data,
        n_oscillators,
    )
    return dstate
