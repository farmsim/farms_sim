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
            control,
            n_iterations
    ):
        """Default amphibious newtwork parameters"""
        state = OscillatorNetworkState.from_initial_state(
            control.network.state_init(),
            n_iterations,
        )
        oscillators = Oscillators.from_options(
            control.network,
        )
        contacts = ContactsArray.from_names(
            control.sensors.contacts,
            n_iterations,
        )
        oscillators_map = {
            name: osc_i
            for osc_i, name in enumerate(oscillators.names)
        }
        contacts_map = {
            name: sens_i
            for sens_i, name in enumerate(contacts.names)
        }
        network = NetworkParameters(
            drives=DriveArray.from_initial_drive(
                control.network.drives_init(),
                n_iterations,
            ),
            oscillators=oscillators,
            osc_connectivity=OscillatorConnectivity.from_connectivity(
                control.network.osc2osc,
                map1=oscillators_map,
                map2=oscillators_map,
            ),
            drive_connectivity=ConnectivityCy(
                control.network.drive2osc,
            ),
            joints_connectivity=JointsConnectivity.from_connectivity(
                control.network.joint2osc,
                map1=oscillators_map,
            ),
            contacts_connectivity=ContactsConnectivity.from_connectivity(
                control.network.contact2osc,
                map1=oscillators_map,
                map2=contacts_map,
            ),
            hydro_connectivity=HydroConnectivity.from_connectivity(
                control.network.hydro2osc,
                map1=oscillators_map,
            ),
        )
        joints = JointsArray.from_options(control)
        sensors = SensorsData(
            contacts=contacts,
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
