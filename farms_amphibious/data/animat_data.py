"""Animat data"""

import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
from .animat_data_cy import (
    AnimatDataCy,
    NetworkParametersCy,
    OscillatorNetworkStateCy,
    OscillatorArrayCy,
    OscillatorConnectivityCy,
    ContactConnectivityCy,
    HydroConnectivityCy,
    JointsArrayCy,
    SensorsDataCy,
    ContactsArrayCy,
    ProprioceptionArrayCy,
    GpsArrayCy,
    HydrodynamicsArrayCy
)


DTYPE = np.float64
ITYPE = np.uintc


def to_array(array, iteration=None):
    """To array or None"""
    if array is not None:
        array = np.array(array)
        if iteration is not None:
            array = array[:iteration]
    return array


class AnimatData(AnimatDataCy):
    """Animat data"""

    @classmethod
    def from_dict(cls, dictionary, n_oscillators=0):
        """Load data from dictionary"""
        return cls(
            state=OscillatorNetworkState(dictionary['state'], n_oscillators),
            network=NetworkParameters.from_dict(dictionary['network']),
            joints=JointsArray(dictionary['joints']),
            sensors=SensorsData.from_dict(dictionary['sensors']),
        )

    @classmethod
    def from_file(cls, filename, n_oscillators=0):
        """From file"""
        return cls.from_dict(dd.io.load(filename), n_oscillators)

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        return {
            'state': to_array(self.state.array),
            'network': self.network.to_dict(iteration),
            'joints': to_array(self.joints.array),
            'sensors': self.sensors.to_dict(iteration),
        }

    def to_file(self, filename, iteration=None):
        """Save data to file"""
        dd.io.save(filename, self.to_dict(iteration))

    def plot(self, times):
        """Plot"""
        self.state.plot(times)
        self.sensors.plot(times)


class NetworkParameters(NetworkParametersCy):
    """Network parameter"""

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            oscillators=OscillatorArray(
                dictionary['oscillators']
            ),
            osc_connectivity=OscillatorConnectivity.from_dict(
                dictionary['osc_connectivity']
            ),
            contacts_connectivity=ContactConnectivity.from_dict(
                dictionary['contacts_connectivity']
            ),
            hydro_connectivity=HydroConnectivity.from_dict(
                dictionary['hydro_connectivity']
            ),
        )

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {
            'oscillators': to_array(self.oscillators.array),
            'osc_connectivity': self.osc_connectivity.to_dict(),
            'contacts_connectivity': self.contacts_connectivity.to_dict(),
            'hydro_connectivity': self.hydro_connectivity.to_dict(),
        }


class OscillatorNetworkState(OscillatorNetworkStateCy):
    """Network state"""

    def plot(self, times):
        """Plot"""
        self.plot_phases(times)
        self.plot_amplitudes(times)

    def plot_phases(self, times):
        """Plot phases"""
        plt.figure('Network state phases')
        for data in np.transpose(self.phases_all()):
            plt.plot(times, data[:len(times)])
        plt.xlabel('Times [s]')
        plt.ylabel('Phases [rad]')
        plt.grid(True)

    def plot_amplitudes(self, times):
        """Plot amplitudes"""
        plt.figure('Network state amplitudes')
        for data in np.transpose(self.amplitudes_all()):
            plt.plot(times, data[:len(times)])
        plt.xlabel('Times [s]')
        plt.ylabel('Amplitudes')
        plt.grid(True)


class OscillatorArray(OscillatorArrayCy):
    """Oscillator array"""


class OscillatorConnectivity(OscillatorConnectivityCy):
    """Connectivity array"""

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            connections=dictionary['connections'],
            weights=dictionary['weights'],
            desired_phases=dictionary['desired_phases'],
        )

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {
            'connections': to_array(self.connections.array),
            'weights': to_array(self.weights.array),
            'desired_phases': to_array(self.desired_phases.array),
        }


class ContactConnectivity(ContactConnectivityCy):
    """Connectivity array"""

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            connections=dictionary['connections'],
            weights=dictionary['weights'],
        )

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        return {
            'connections': to_array(self.connections.array),
            'weights': to_array(self.weights.array),
        }

    @classmethod
    def from_connectivity(cls, connectivity):
        """From connectivity"""
        connections = [
            [connection['in'], connection['out']]
            for connection in connectivity
        ]
        weights = [
            connection['weight']
            for connection in connectivity
        ]
        return cls(
            np.array(connections, dtype=ITYPE),
            np.array(weights, dtype=DTYPE),
        )


class HydroConnectivity(HydroConnectivityCy):
    """Connectivity array"""

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            connections=dictionary['connections'],
            frequency=dictionary['frequency'],
            amplitude=dictionary['amplitude'],
        )

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {
            'connections': to_array(self.connections.array),
            'frequency': to_array(self.frequency.array),
            'amplitude': to_array(self.amplitude.array),
        }

    @classmethod
    def from_connectivity(cls, connectivity):
        """From connectivity"""
        connections = [
            [connection['in'], connection['out']]
            for connection in connectivity
        ]
        weights_frequency = [
            connection['weight_frequency']
            for connection in connectivity
        ]
        weights_amplitude = [
            connection['weight_amplitude']
            for connection in connectivity
        ]
        return cls(
            connections=np.array(connections, dtype=ITYPE),
            frequency=np.array(weights_frequency, dtype=DTYPE),
            amplitude=np.array(weights_amplitude, dtype=DTYPE),
        )


class JointsArray(JointsArrayCy):
    """Oscillator array"""


class SensorsData(SensorsDataCy):
    """SensorsData"""

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            contacts=ContactsArray(dictionary['contacts']),
            proprioception=ProprioceptionArray(dictionary['proprioception']),
            gps=GpsArray(dictionary['gps']),
            hydrodynamics=HydrodynamicsArray(dictionary['hydrodynamics']),
        )

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        return {
            'contacts': to_array(self.contacts.array, iteration),
            'proprioception': to_array(self.proprioception.array, iteration),
            'gps': to_array(self.gps.array, iteration),
            'hydrodynamics': to_array(self.hydrodynamics.array, iteration),
        }

    def plot(self, times):
        """Plot"""
        self.contacts.plot(times)
        self.proprioception.plot(times)
        self.gps.plot(times)
        self.hydrodynamics.plot(times)


class ContactsArray(ContactsArrayCy):
    """Sensor array"""

    def plot(self, times):
        """Plot"""
        self.plot_ground_reaction_forces(times)
        self.plot_friction_forces(times)
        for ori in range(3):
            self.plot_friction_forces_ori(times, ori=ori)
        self.plot_total_forces(times)

    def plot_ground_reaction_forces(self, times):
        """Plot ground reaction forces"""
        plt.figure('Ground reaction forces')
        for sensor_i in range(self.size(1)):
            data = np.asarray(self.reaction_all(sensor_i))
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1)[:len(times)],
                label='Leg_{}'.format(sensor_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)

    def plot_friction_forces(self, times):
        """Plot friction forces"""
        plt.figure('Friction forces')
        for sensor_i in range(self.size(1)):
            data = np.asarray(self.friction_all(sensor_i))
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1)[:len(times)],
                label='Leg_{}'.format(sensor_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)

    def plot_friction_forces_ori(self, times, ori):
        """Plot friction forces"""
        plt.figure('Friction forces (ori={})'.format(ori))
        for sensor_i in range(self.size(1)):
            data = np.asarray(self.friction_all(sensor_i))
            plt.plot(
                times,
                data[:len(times), ori],
                label='Leg_{}'.format(sensor_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)

    def plot_total_forces(self, times):
        """Plot contact forces"""
        plt.figure('Contact total forces')
        for sensor_i in range(self.size(1)):
            data = np.asarray(self.total_all(sensor_i))
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1)[:len(times)],
                label='Leg_{}'.format(sensor_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)


class ProprioceptionArray(ProprioceptionArrayCy):
    """Proprioception array"""

    def plot(self, times):
        """Plot"""
        self.plot_positions(times)
        self.plot_velocities(times)
        self.plot_forces(times)
        self.plot_torques(times)
        self.plot_motor_torques(times)
        self.plot_active_torques(times)
        self.plot_spring_torques(times)
        self.plot_damping_torques(times)

    def plot_positions(self, times):
        """Plot ground reaction forces"""
        plt.figure('Joints positions')
        for joint_i in range(self.size(1)):
            plt.plot(
                times,
                np.asarray(self.positions_all())[:len(times), joint_i],
                label='Joint_{}'.format(joint_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint position [rad]')
        plt.grid(True)

    def plot_velocities(self, times):
        """Plot ground reaction forces"""
        plt.figure('Joints velocities')
        for joint_i in range(self.size(1)):
            plt.plot(
                times,
                np.asarray(self.velocities_all())[:len(times), joint_i],
                label='Joint_{}'.format(joint_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint velocity [rad/s]')
        plt.grid(True)

    def plot_forces(self, times):
        """Plot ground reaction forces"""
        plt.figure('Joints forces')
        for joint_i in range(self.size(1)):
            data = np.linalg.norm(np.asarray(self.forces_all()), axis=-1)
            plt.plot(
                times,
                data[:len(times), joint_i],
                label='Joint_{}'.format(joint_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint force [N]')
        plt.grid(True)

    def plot_torques(self, times):
        """Plot ground reaction torques"""
        plt.figure('Joints torques')
        for joint_i in range(self.size(1)):
            data = np.linalg.norm(np.asarray(self.torques_all()), axis=-1)
            plt.plot(
                times,
                data[:len(times), joint_i],
                label='Joint_{}'.format(joint_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint torque [Nm]')
        plt.grid(True)

    def plot_motor_torques(self, times):
        """Plot ground reaction forces"""
        plt.figure('Joints motor torques')
        for joint_i in range(self.size(1)):
            plt.plot(
                times,
                np.asarray(self.motor_torques())[:len(times), joint_i],
                label='Joint_{}'.format(joint_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint torque [Nm]')
        plt.grid(True)

    def plot_active_torques(self, times):
        """Plot joints active torques"""
        plt.figure('Joints active torques')
        for joint_i in range(self.size(1)):
            plt.plot(
                times,
                np.asarray(self.active_torques())[:len(times), joint_i],
                label='Joint_{}'.format(joint_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint torque [Nm]')
        plt.grid(True)

    def plot_spring_torques(self, times):
        """Plot joints spring torques"""
        plt.figure('Joints spring torques')
        for joint_i in range(self.size(1)):
            plt.plot(
                times,
                np.asarray(self.spring_torques())[:len(times), joint_i],
                label='Joint_{}'.format(joint_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint torque [Nm]')
        plt.grid(True)

    def plot_damping_torques(self, times):
        """Plot joints damping torques"""
        plt.figure('Joints damping torques')
        for joint_i in range(self.size(1)):
            plt.plot(
                times,
                np.asarray(self.damping_torques())[:len(times), joint_i],
                label='Joint_{}'.format(joint_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint torque [Nm]')
        plt.grid(True)


class GpsArray(GpsArrayCy):
    """Gps array"""

    def plot(self, times):
        """Plot"""
        self.plot_base_position(times, xaxis=0, yaxis=1)
        self.plot_base_velocity(times)

    def plot_base_position(self, times, xaxis=0, yaxis=1):
        """Plot"""
        plt.figure('GPS position')
        for link_i in range(self.size(1)):
            data = np.asarray(self.urdf_positions())[:len(times), link_i]
            plt.plot(
                data[:, xaxis],
                data[:, yaxis],
                label='Link_{}'.format(link_i)
            )
        plt.legend()
        plt.xlabel('Position [m]')
        plt.ylabel('Position [m]')
        plt.axis('equal')
        plt.grid(True)

    def plot_base_velocity(self, times):
        """Plot"""
        plt.figure('GPS velocities')
        for link_i in range(self.size(1)):
            data = np.asarray(self.com_lin_velocities())[:len(times), link_i]
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1),
                label='Link_{}'.format(link_i)
            )
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.grid(True)


class HydrodynamicsArray(HydrodynamicsArrayCy):
    """Hydrodynamics array"""

    def plot(self, times):
        """Plot"""
        self.plot_forces(times)
        self.plot_torques(times)

    def plot_forces(self, times):
        """Plot"""
        plt.figure('Hydrodynamic forces')
        for link_i in range(self.size(1)):
            data = np.asarray(self.forces())[:len(times), link_i]
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1),
                label='Link_{}'.format(link_i)
            )
        plt.xlabel('Time [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)

    def plot_torques(self, times):
        """Plot"""
        plt.figure('Hydrodynamic torques')
        for link_i in range(self.size(1)):
            data = np.asarray(self.torques())[:len(times), link_i]
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1),
                label='Link_{}'.format(link_i)
            )
        plt.xlabel('Time [s]')
        plt.ylabel('Torques [Nm]')
        plt.grid(True)
