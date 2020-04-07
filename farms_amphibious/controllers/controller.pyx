"""Cython code"""

import time
# import numpy as np

cimport cython
cimport numpy as np

from libc.math cimport sin, fabs  # cos,
# from libc.stdlib cimport malloc, free
# from cython.parallel import prange


cpdef void ode_dphase(
    CTYPEv1 state,
    CTYPEv1 dstate,
    OscillatorArrayCy oscillators,
    ConnectivityArrayCy connectivity,
) nogil:
    """Oscillator phase ODE

    d_theta = omega + sum amplitude_j*weight*sin(phase_j - phase_i - phase_bias)

    """
    cdef unsigned int i, i0, i1
    cdef unsigned int n_oscillators = oscillators.size[1]
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
    CTYPEv1 state,
    CTYPEv1 dstate,
    OscillatorArrayCy oscillators,
) nogil:
    """Oscillator amplitude ODE

    d_amplitude = rate*(nominal_amplitude - amplitude)

    """
    cdef unsigned int i
    cdef unsigned int n_oscillators = oscillators.size[1]
    for i in range(n_oscillators):  # , nogil=True):
        # rate*(nominal_amplitude - amplitude)
        dstate[n_oscillators+i] = oscillators.array[1][i]*(
            oscillators.array[2][i] - state[n_oscillators+i]
        )


cpdef void ode_contacts(
    unsigned int iteration,
    CTYPEv1 state,
    CTYPEv1 dstate,
    ContactsArrayCy contacts,
    ConnectivityArrayCy contacts_connectivity,
) nogil:
    """Sensory feedback - Contacts

    Can affect d_phase and d_amplitude

    """
    cdef CTYPE contact_force
    cdef unsigned int i, i0, i1
    for i in range(contacts_connectivity.size[0]):
        i0 = <unsigned int> (
            contacts_connectivity.array[i][0] + 0.5
        )
        i1 = <unsigned int> (
            contacts_connectivity.array[i][1] + 0.5
        )
        # contact_weight*contact_force
        # contact_force = (
        #     contacts.array[iteration][i1][0]**2
        #     + contacts.array[iteration][i1][1]**2
        #     + contacts.array[iteration][i1][2]**2
        # )**0.5
        contact_force = fabs(contacts.array[iteration][i1][2])
        dstate[i0] += (
            contacts_connectivity.array[i][2]
            *(10*contact_force/(1+10*contact_force))  # Saturation
            # *cos(state[i0])
            # *sin(state[i0])  # For Tegotae
        )


cpdef void ode_hydro(
    unsigned int iteration,
    CTYPEv1 state,
    CTYPEv1 dstate,
    HydrodynamicsArrayCy hydrodynamics,
    ConnectivityArrayCy hydro_connectivity,
    unsigned int n_oscillators,
) nogil:
    """Sensory feedback - Hydrodynamics

    Can affect d_phase and d_amplitude

    """
    cdef CTYPE hydro_force
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
    CTYPEv1 state,
    CTYPEv1 dstate,
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


cpdef CTYPEv1 ode_oscillators_sparse(
    CTYPE time,
    CTYPEv1 state,
    AnimatDataCy data,
    NetworkParametersCy network,
) nogil:
    """Complete CPG network ODE"""
    ode_dphase(
        state=state,
        dstate=data.state.array[data.iteration][1],
        oscillators=data.network.oscillators,
        connectivity=data.network.connectivity,
    )
    ode_damplitude(
        state=state,
        dstate=data.state.array[data.iteration][1],
        oscillators=data.network.oscillators,
    )
    ode_contacts(
        iteration=data.iteration,
        state=state,
        dstate=data.state.array[data.iteration][1],
        contacts=data.sensors.contacts,
        contacts_connectivity=data.network.contacts_connectivity,
    )
    ode_hydro(
        iteration=data.iteration,
        state=state,
        dstate=data.state.array[data.iteration][1],
        hydrodynamics=data.sensors.hydrodynamics,
        hydro_connectivity=data.network.hydro_connectivity,
        n_oscillators=data.network.oscillators.size[1],
    )
    ode_joints(
        state=state,
        dstate=data.state.array[data.iteration][1],
        joints=data.joints,
        n_oscillators=data.network.oscillators.size[1],
    )
    return data.state.array[data.iteration][1]


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
) nogil:
    """Complete CPG network ODE"""
    cdef unsigned int i, state_size = state.shape[0]
    cdef CTYPE rk_const = 1.0/6.0
    k1 = ode_oscillators_sparse(
        time=time,
        state=state,
        data=data,
        network=network,
    )
    for i in range(state_size):
        k1[i] = timestep*k1[i]
        state_out[i] = state[i] + 0.5*k1[i]
    k2 = ode_oscillators_sparse(
        time=time+0.5*timestep,
        state=state_out,
        data=data,
        network=network,
    )
    for i in range(state_size):
        k2[i] = timestep*k2[i]
        state_out[i] = state[i] + 0.5*k2[i]
    k3 = ode_oscillators_sparse(
        time=time + 0.5*timestep,
        state=state_out,
        data=data,
        network=network,
    )
    for i in range(state_size):
        k3[i] = timestep*k3[i]
        state_out[i] = state[i] + k3[i]
    k4 = ode_oscillators_sparse(
        time=time + timestep,
        state=state_out,
        data=data,
        network=network,
    )
    for i in range(state_size):
        k4[i] = timestep*k4[i]
        state_out[i] = state[i] + rk_const*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
