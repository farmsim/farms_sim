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
    JointsConnectivity,
    ContactsConnectivity,
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
            joints_connectivity=JointsConnectivity.from_connectivity(
                control.network.joint2osc
            ),
            contacts_connectivity=ContactsConnectivity.from_connectivity(
                control.network.contact2osc
            ),
            hydro_connectivity=HydroConnectivity.from_connectivity(
                control.network.hydro2osc,
            ),
        )
        joints = JointsArray.from_options(control)
        sensors = SensorsData(
            contacts=ContactsArray.from_names(
                control.sensors.contacts,
                n_iterations,
            ),
            proprioception=ProprioceptionArray.from_names(
                control.sensors.joints,
                n_iterations,
            ),
            gps=GpsArray.from_names(
                control.sensors.gps,
                n_iterations,
            ),
            hydrodynamics=HydrodynamicsArray.from_names(
                control.sensors.hydrodynamics,
                n_iterations,
            )
        )
        return cls(state, network, joints, sensors)
