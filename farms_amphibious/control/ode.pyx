"""Cython code"""

# import time
# import numpy as np
# cimport numpy as np

# cimport cython
# from cython.parallel import prange

from libc.math cimport sin, cos, fabs
# from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

from farms_data.amphibious.animat_data_cy cimport ConnectionType


cdef inline DTYPE phase(
    DTYPEv1 state,
    unsigned int index
) nogil:
    """Phase"""
    return state[index]


cdef inline DTYPE amplitude(
    DTYPEv1 state,
    unsigned int index,
    unsigned int n_oscillators,
) nogil:
    """Amplitude"""
    return state[index+n_oscillators]


cdef inline DTYPE joint_offset(
    DTYPEv1 state,
    unsigned int index,
    unsigned int n_oscillators,
) nogil:
    """Joint offset"""
    return state[index+2*n_oscillators]


cdef inline DTYPE saturation(DTYPE value, DTYPE multiplier) nogil:
    """Saturation from 0 to 1"""
    return multiplier*value/(1+multiplier*value)


cpdef inline void ode_dphase(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    DriveArrayCy drives,
    OscillatorsCy oscillators,
    OscillatorsConnectivityCy connectivity,
) nogil:
    """Oscillator phase ODE

    d_theta = (
        omega*(1 + omega_mod_amp*cos(theta + omega_mod_phase))
        + sum amplitude_j*weight*sin(phase_j - phase_i - phase_bias)
    )

    """
    cdef unsigned int i, i0, i1, n_oscillators = oscillators.c_n_oscillators()
    for i in range(n_oscillators):
        # Intrinsic frequency
        dstate[i] = oscillators.c_angular_frequency(i, drives.c_speed(iteration))
        if oscillators.c_modular_amplitudes(i) > 1e-3:
            dstate[i] *= (
                1 + oscillators.c_modular_amplitudes(i)*cos(
                    phase(state, i) + oscillators.c_modular_phases(i)
                )
            )
    for i in range(connectivity.c_n_connections()):
        # Neural couplings
        i0 = connectivity.connections.array[i, 0]
        i1 = connectivity.connections.array[i, 1]
        dstate[i0] += state[n_oscillators+i1]*connectivity.c_weight(i)*sin(
            phase(state, i1) - phase(state, i0)
            - connectivity.c_desired_phase(i)
        )


cpdef inline void ode_damplitude(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    DriveArrayCy drives,
    OscillatorsCy oscillators,
) nogil:
    """Oscillator amplitude ODE

    d_amplitude = rate*(nominal_amplitude - amplitude)

    """
    cdef unsigned int i, n_oscillators = oscillators.c_n_oscillators()
    for i in range(n_oscillators):  # , nogil=True):
        # rate*(nominal_amplitude - amplitude)
        dstate[n_oscillators+i] = oscillators.c_rate(i)*(
            oscillators.c_nominal_amplitude(i, drives.c_speed(iteration))
            - amplitude(state, i, n_oscillators)
        )


cpdef inline void ode_stretch(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    JointSensorArrayCy joints,
    JointsConnectivityCy joints_connectivity,
    unsigned int n_oscillators,
) nogil:
    """Sensory feedback - Stretch

    Can affect d_phase

    """
    cdef unsigned int i, i0, i1, connection_type
    for i in range(joints_connectivity.c_n_connections()):
        i0 = joints_connectivity.connections.array[i, 0]
        i1 = joints_connectivity.connections.array[i, 1]
        connection_type = joints_connectivity.connections.array[i, 2]
        if connection_type == ConnectionType.STRETCH2FREQ:
            # stretch_weight*joint_position  # *sin(phase)
            dstate[i0] += (
                joints_connectivity.c_weight(i)
                *joints.position_cy(iteration, i1)
                # *sin(state[i0])  # For Tegotae
            )
        elif connection_type == ConnectionType.STRETCH2AMP:
            # stretch_weight*joint_position  # *sin(phase)
            dstate[n_oscillators+i0] += (
                joints_connectivity.c_weight(i)
                *joints.position_cy(iteration, i1)
                # *sin(state[i0])  # For Tegotae
            )
        else:
            printf(
                'Joint connection %i of type %i is incorrect'
                ', should be %i or %i\n',
                i,
                connection_type,
                ConnectionType.STRETCH2FREQ,
                ConnectionType.STRETCH2AMP,
            )


cpdef inline void ode_contacts(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    ContactsArrayCy contacts,
    ContactsConnectivityCy contacts_connectivity,
) nogil:
    """Sensory feedback - Contacts

    Can affect d_phase and d_amplitude

    """
    cdef DTYPE contact_reaction
    cdef unsigned int i, i0, i1, connection_type
    for i in range(contacts_connectivity.c_n_connections()):
        i0 = contacts_connectivity.connections.array[i, 0]
        i1 = contacts_connectivity.connections.array[i, 1]
        connection_type = contacts_connectivity.connections.array[i, 2]
        contact_reaction = fabs(contacts.c_reaction_z(iteration, i1))
        if connection_type == ConnectionType.REACTION2FREQ:
            dstate[i0] += (
                contacts_connectivity.c_weight(i)
                *saturation(contact_reaction, 10)  # Saturation
            )
        elif connection_type == ConnectionType.REACTION2FREQTEGOTAE:
            dstate[i0] += (
                contacts_connectivity.c_weight(i)
                *saturation(contact_reaction, 10)  # Saturation
                # *cos(state[i0])
                *sin(state[i0])  # For Tegotae
            )


cpdef inline void ode_contacts_tegotae(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    ContactsArrayCy contacts,
    ContactsConnectivityCy contacts_connectivity,
) nogil:
    """Sensory feedback - Contacts

    Can affect d_phase and d_amplitude

    """
    cdef DTYPE contact_reaction
    cdef unsigned int i, i0, i1
    for i in range(contacts_connectivity.c_n_connections()):
        i0 = contacts_connectivity.connections.array[i, 0]
        i1 = contacts_connectivity.connections.array[i, 1]
        # contact_weight*contact_reaction
        # contact_reaction = (
        #     contacts.array[iteration, i1, 0]**2
        #     + contacts.array[iteration, i1, 1]**2
        #     + contacts.array[iteration, i1, 2]**2
        # )**0.5
        contact_reaction = fabs(contacts.c_reaction_z(iteration, i1))
        dstate[i0] += (
            contacts_connectivity.c_weight(i)
            *saturation(contact_reaction, 10)  # Saturation
            *sin(state[i0])  # For Tegotae
        )


cpdef inline void ode_hydro(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    HydrodynamicsArrayCy hydrodynamics,
    HydroConnectivityCy hydro_connectivity,
    unsigned int n_oscillators,
) nogil:
    """Sensory feedback - Hydrodynamics

    Can affect d_phase and d_amplitude

    """
    cdef DTYPE hydro_force
    cdef unsigned int i, i0, i1, connection_type
    for i in range(hydro_connectivity.c_n_connections()):
        i0 = hydro_connectivity.connections.array[i, 0]
        i1 = hydro_connectivity.connections.array[i, 1]
        connection_type = hydro_connectivity.connections.array[i, 2]
        hydro_force = fabs(hydrodynamics.c_force_y(iteration, i1))
        if connection_type == ConnectionType.LATERAL2FREQ:
            # dfrequency += hydro_weight*hydro_force
            dstate[i0] += (
                hydro_connectivity.c_weights(i)*hydro_force
            )
        elif connection_type == ConnectionType.LATERAL2AMP:
            # damplitude += hydro_weight*hydro_force
            dstate[n_oscillators+i0] += (
                hydro_connectivity.c_weights(i)*hydro_force
            )
        else:
            printf(
                'Hydrodynamics connection %i of type %i is incorrect'
                ', should be %i or %i instead\n',
                i,
                connection_type,
                ConnectionType.LATERAL2FREQ,
                ConnectionType.LATERAL2AMP,
            )


cpdef inline void ode_joints(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    DriveArrayCy drives,
    JointsControlArrayCy joints,
    unsigned int n_oscillators,
) nogil:
    """Joints offset

    d_joints_offset = rate*(joints_offset_desired - joints_offset)

    """
    cdef unsigned int joint_i, n_joints = joints.c_n_joints()
    for joint_i in range(n_joints):
        # rate*(joints_offset_desired - joints_offset)
        dstate[2*n_oscillators+joint_i] = joints.c_rate(joint_i)*(
            joints.c_offset_desired(
                joint_i,
                drives.c_turn(iteration),
                drives.c_speed(iteration),
            ) - joint_offset(state, joint_i, n_oscillators)
        )


## ODEs

cpdef inline DTYPEv1 ode_oscillators_sparse(
    DTYPE time,
    DTYPEv1 state,
    DTYPEv1 dstate,
    unsigned int iteration,
    AnimatDataCy data,
) nogil:
    """Complete CPG network ODE"""
    ode_dphase(
        iteration=iteration,
        state=state,
        dstate=dstate,
        drives=data.network.drives,
        oscillators=data.network.oscillators,
        connectivity=data.network.osc_connectivity,
    )
    ode_damplitude(
        iteration=iteration,
        state=state,
        dstate=dstate,
        drives=data.network.drives,
        oscillators=data.network.oscillators,
    )
    ode_stretch(
        iteration=iteration,
        state=state,
        dstate=dstate,
        joints=data.sensors.joints,
        joints_connectivity=data.network.joints_connectivity,
    )
    ode_contacts(
        iteration=iteration,
        state=state,
        dstate=dstate,
        contacts=data.sensors.contacts,
        contacts_connectivity=data.network.contacts_connectivity,
    )
    ode_hydro(
        iteration=iteration,
        state=state,
        dstate=dstate,
        hydrodynamics=data.sensors.hydrodynamics,
        hydro_connectivity=data.network.hydro_connectivity,
        n_oscillators=data.network.oscillators.c_n_oscillators(),
    )
    ode_joints(
        iteration=iteration,
        state=state,
        dstate=dstate,
        drives=data.network.drives,
        joints=data.joints,
        n_oscillators=data.network.oscillators.c_n_oscillators(),
    )
    # data.network.drives.array[iteration+1] = data.network.drives.array[iteration]
    return dstate


cpdef inline DTYPEv1 ode_oscillators_sparse_no_sensors(
    DTYPE time,
    DTYPEv1 state,
    DTYPEv1 dstate,
    unsigned int iteration,
    AnimatDataCy data,
) nogil:
    """CPG network ODE using no sensors"""
    ode_dphase(
        iteration=iteration,
        state=state,
        dstate=dstate,
        drives=data.network.drives,
        oscillators=data.network.oscillators,
        connectivity=data.network.osc_connectivity,
    )
    ode_damplitude(
        iteration=iteration,
        state=state,
        dstate=dstate,
        drives=data.network.drives,
        oscillators=data.network.oscillators,
    )
    ode_joints(
        iteration=iteration,
        state=state,
        dstate=dstate,
        drives=data.network.drives,
        joints=data.joints,
        n_oscillators=data.network.oscillators.c_n_oscillators(),
    )
    # data.network.drives.array[iteration+1] = data.network.drives.array[iteration]
    return dstate


cpdef inline DTYPEv1 ode_oscillators_sparse_tegotae(
    DTYPE time,
    DTYPEv1 state,
    DTYPEv1 dstate,
    unsigned int iteration,
    AnimatDataCy data,
) nogil:
    """CPG network ODE using Tegotae"""
    ode_dphase(
        iteration=iteration,
        state=state,
        dstate=dstate,
        drives=data.network.drives,
        oscillators=data.network.oscillators,
        connectivity=data.network.osc_connectivity,
    )
    ode_damplitude(
        iteration=iteration,
        state=state,
        dstate=dstate,
        drives=data.network.drives,
        oscillators=data.network.oscillators,
    )
    ode_contacts_tegotae(
        iteration=iteration,
        state=state,
        dstate=dstate,
        contacts=data.sensors.contacts,
        contacts_connectivity=data.network.contacts_connectivity,
    )
    ode_joints(
        iteration=iteration,
        state=state,
        dstate=dstate,
        drives=data.network.drives,
        joints=data.joints,
        n_oscillators=data.network.oscillators.c_n_oscillators(),
    )
    # data.network.drives.array[iteration+1] = data.network.drives.array[iteration]
    return dstate
