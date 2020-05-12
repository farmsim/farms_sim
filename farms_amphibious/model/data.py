"""Animat data"""

from farms_bullet.data.data import (
    SensorsData,
    ContactsArray,
    ProprioceptionArray,
    GpsArray,
    HydrodynamicsArray,
)
from ..data.animat_data_cy import ConnectivityCy
from ..data.animat_data import (
    OscillatorNetworkState,
    AnimatData,
    NetworkParameters,
    DriveArray,
    Oscillators,
    OscillatorConnectivity,
    JointConnectivity,
    ContactConnectivity,
    HydroConnectivity,
    JointsArray,
)


class AmphibiousData(AnimatData):
    """Amphibious network parameter"""

    @classmethod
    def from_options(
            cls,
            initial_drives,
            initial_state,
            morphology,
            control,
            n_iterations
    ):
        """Default amphibious newtwork parameters"""
        state = OscillatorNetworkState.from_initial_state(
            initial_state,
            n_iterations,
        )
        network = NetworkParameters(
            drives=DriveArray.from_initial_drive(
                initial_drives,
                n_iterations,
            ),
            oscillators=Oscillators.from_options(
                control.network,
            ),
            osc_connectivity=OscillatorConnectivity.from_connectivity(
                control.network.osc2osc
            ),
            drive_connectivity=ConnectivityCy(control.network.drive2osc),
            joint_connectivity=JointConnectivity.from_connectivity(
                control.network.joint2osc
            ),
            contacts_connectivity=ContactConnectivity.from_connectivity(
                control.network.contact2osc
            ),
            hydro_connectivity=HydroConnectivity.from_connectivity(
                control.network.hydro2osc,
            ),
        )
        joints = JointsArray.from_options(control.joints)
        sensors = SensorsData(
            contacts=ContactsArray.from_size(
                morphology.n_legs,
                n_iterations,
            ),
            proprioception=ProprioceptionArray.from_size(
                morphology.n_joints(),
                n_iterations,
            ),
            gps=GpsArray.from_size(
                morphology.n_links(),
                n_iterations,
            ),
            hydrodynamics=HydrodynamicsArray.from_size(
                morphology.n_links_body(),
                n_iterations,
            )
        )
        return cls(state, network, joints, sensors)
