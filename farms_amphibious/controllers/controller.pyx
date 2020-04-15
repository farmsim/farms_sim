"""Cython code"""

import time
# import numpy as np

cimport cython
cimport numpy as np

from libc.math cimport sin, fabs  # cos,
# from libc.stdlib cimport malloc, free
# from cython.parallel import prange


cdef inline CTYPE phase(
    CTYPEv1 state,
    unsigned int index
) nogil:
    """Phase"""
    return state[index]


cdef inline CTYPE amplitude(
    CTYPEv1 state,
    unsigned int index,
    unsigned int n_oscillators,
) nogil:
    """Amplitude"""
    return state[index+n_oscillators]


cdef inline CTYPE joint_offset(
    CTYPEv1 state,
    unsigned int index,
    unsigned int n_oscillators,
) nogil:
    """Joint offset"""
    return state[index+2*n_oscillators]


cdef inline CTYPE saturation(CTYPE value, CTYPE multiplier) nogil:
    """Saturation from 0 to 1"""
    return multiplier*value/(1+multiplier*value)


cpdef inline void ode_dphase(
    CTYPEv1 state,
    CTYPEv1 dstate,
    OscillatorArrayCy oscillators,
    OscillatorConnectivityCy connectivity,
) nogil:
    """Oscillator phase ODE

    d_theta = omega + sum amplitude_j*weight*sin(phase_j - phase_i - phase_bias)

    """
    cdef unsigned int i, i0, i1, n_oscillators = oscillators.c_n_oscillators()
    for i in range(n_oscillators):  # , nogil=True):
        # Intrinsic frequency
        dstate[i] = oscillators.c_angular_frequency(i)
    for i in range(connectivity.c_n_connections()):
        i0 = connectivity.connections.array[i][0]
        i1 = connectivity.connections.array[i][1]
        dstate[i0] += state[n_oscillators+i1]*connectivity.c_weight(i)*sin(
            phase(state, i1) - phase(state, i0)
            - connectivity.c_desired_phase(i)
        )


cpdef inline void ode_damplitude(
    CTYPEv1 state,
    CTYPEv1 dstate,
    OscillatorArrayCy oscillators,
) nogil:
    """Oscillator amplitude ODE

    d_amplitude = rate*(nominal_amplitude - amplitude)

    """
    cdef unsigned int i, n_oscillators = oscillators.c_n_oscillators()
    for i in range(n_oscillators):  # , nogil=True):
        # rate*(nominal_amplitude - amplitude)
        dstate[n_oscillators+i] = oscillators.c_rate(i)*(
            oscillators.c_nominal_amplitude(i)
            - amplitude(state, i, n_oscillators)
        )


cpdef inline void ode_contacts(
    unsigned int iteration,
    CTYPEv1 state,
    CTYPEv1 dstate,
    ContactsArrayCy contacts,
    ContactConnectivityCy contacts_connectivity,
) nogil:
    """Sensory feedback - Contacts

    Can affect d_phase and d_amplitude

    """
    cdef CTYPE contact_force
    cdef unsigned int i, i0, i1
    for i in range(contacts_connectivity.c_n_connections()):
        i0 = contacts_connectivity.connections.array[i][0]
        i1 = contacts_connectivity.connections.array[i][1]
        # contact_weight*contact_force
        # contact_force = (
        #     contacts.array[iteration][i1][0]**2
        #     + contacts.array[iteration][i1][1]**2
        #     + contacts.array[iteration][i1][2]**2
        # )**0.5
        contact_force = fabs(contacts.c_force_z(iteration, i1))
        dstate[i0] += (
            contacts_connectivity.c_weight(i)
            *saturation(contact_force, 10)  # Saturation
            # *cos(state[i0])
            # *sin(state[i0])  # For Tegotae
        )


cpdef inline void ode_hydro(
    unsigned int iteration,
    CTYPEv1 state,
    CTYPEv1 dstate,
    HydrodynamicsArrayCy hydrodynamics,
    HydroConnectivityCy hydro_connectivity,
    unsigned int n_oscillators,
) nogil:
    """Sensory feedback - Hydrodynamics

    Can affect d_phase and d_amplitude

    """
    cdef CTYPE hydro_force
    cdef unsigned int i, i0, i1
    for i in range(hydro_connectivity.c_n_connections()):
        i0 = hydro_connectivity.connections.array[i][0]
        i1 = hydro_connectivity.connections.array[i][1]
        hydro_force = fabs(hydrodynamics.c_force_y(iteration, i1))
        # dfrequency += hydro_weight*hydro_force
        dstate[i0] += (
            hydro_connectivity.c_weight_frequency(i)*hydro_force
        )
        # damplitude += hydro_weight*hydro_force
        dstate[n_oscillators+i0] += (
            hydro_connectivity.c_weight_amplitude(i)*hydro_force
        )


cpdef inline void ode_joints(
    CTYPEv1 state,
    CTYPEv1 dstate,
    JointsArrayCy joints,
    unsigned int n_oscillators,
) nogil:
    """Joints offset

    d_joints_offset = rate*(joints_offset_desired - joints_offset)

    """
    cdef unsigned int i, n_joints = joints.c_n_joints()
    for i in range(n_joints):
        # rate*(joints_offset_desired - joints_offset)
        dstate[2*n_oscillators+i] = joints.c_rate(i)*(
            joints.c_offset_desired(i) - joint_offset(state, i, n_oscillators)
        )


cpdef inline CTYPEv1 ode_oscillators_sparse(
    CTYPE time,
    CTYPEv1 state,
    unsigned int iteration,
    AnimatDataCy data,
    NetworkParametersCy network,
) nogil:
    """Complete CPG network ODE"""
    ode_dphase(
        state=state,
        dstate=data.state.array[iteration][1],
        oscillators=data.network.oscillators,
        connectivity=data.network.osc_connectivity,
    )
    ode_damplitude(
        state=state,
        dstate=data.state.array[iteration][1],
        oscillators=data.network.oscillators,
    )
    ode_contacts(
        iteration=iteration,
        state=state,
        dstate=data.state.array[iteration][1],
        contacts=data.sensors.contacts,
        contacts_connectivity=data.network.contacts_connectivity,
    )
    ode_hydro(
        iteration=iteration,
        state=state,
        dstate=data.state.array[iteration][1],
        hydrodynamics=data.sensors.hydrodynamics,
        hydro_connectivity=data.network.hydro_connectivity,
        n_oscillators=data.network.oscillators.c_n_oscillators(),
    )
    ode_joints(
        state=state,
        dstate=data.state.array[iteration][1],
        joints=data.joints,
        n_oscillators=data.network.oscillators.c_n_oscillators(),
    )
    return data.state.array[iteration][1]


# cpdef void rk_oscillators(
#     CTYPE time,
#     CTYPE timestep,
#     CTYPEv1 state,
#     AnimatDataCy data,
#     NetworkParametersCy network,
#     CTYPEv1 k1,
#     CTYPEv1 k2,
#     CTYPEv1 k3,
#     CTYPEv1 k4,
#     CTYPEv1 state_out,
# ) nogil:
#     """Complete CPG network ODE"""
#     cdef unsigned int i, state_size = state.shape[0]
#     cdef CTYPE rk_const = 1.0/6.0
#     k1 = ode_oscillators_sparse(
#         time=time,
#         state=state,
#         data=data,
#         network=network,
#     )
#     for i in range(state_size):
#         k1[i] = timestep*k1[i]
#         state_out[i] = state[i] + 0.5*k1[i]
#     k2 = ode_oscillators_sparse(
#         time=time+0.5*timestep,
#         state=state_out,
#         iteration=data.iteration,
#         data=data,
#         network=network,
#     )
#     for i in range(state_size):
#         k2[i] = timestep*k2[i]
#         state_out[i] = state[i] + 0.5*k2[i]
#     k3 = ode_oscillators_sparse(
#         time=time + 0.5*timestep,
#         state=state_out,
#         data=data,
#         network=network,
#     )
#     for i in range(state_size):
#         k3[i] = timestep*k3[i]
#         state_out[i] = state[i] + k3[i]
#     k4 = ode_oscillators_sparse(
#         time=time + timestep,
#         state=state_out,
#         data=data,
#         network=network,
#     )
#     for i in range(state_size):
#         k4[i] = timestep*k4[i]
#         state_out[i] = state[i] + rk_const*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
