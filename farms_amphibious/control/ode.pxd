"""Cython controller code"""

include 'types.pxd'
from farms_data.sensors.data_cy cimport (
    ContactsArrayCy,
    HydrodynamicsArrayCy,
)
from farms_data.amphibious.animat_data_cy cimport (
    AnimatDataCy,
    NetworkParametersCy,
    DriveArrayCy,
    OscillatorsCy,
    OscillatorsConnectivityCy,
    ContactsConnectivityCy,
    HydroConnectivityCy,
    JointsControlArrayCy,
)


cpdef void ode_dphase(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    DriveArrayCy drives,
    OscillatorsCy oscillators,
    OscillatorsConnectivityCy connectivity,
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
    ContactsConnectivityCy contacts_connectivity,
) nogil


cpdef void ode_contacts_tegotae(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    ContactsArrayCy contacts,
    ContactsConnectivityCy contacts_connectivity,
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
    JointsControlArrayCy joints,
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
