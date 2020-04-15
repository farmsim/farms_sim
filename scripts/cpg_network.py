"""Network"""

import numpy as np
import matplotlib.pyplot as plt

from farms_amphibious.model.options import (
    AmphibiousMorphologyOptions,
    AmphibiousControlOptions,
    AmphibiousDrives,
    AmphibiousNetworkOptions,
    AmphibiousOscillatorOptions,
    AmphibiousConnectivityOptions,
    AmphibiousJointsOptions,
)
from farms_amphibious.model.data import (
    AmphibiousOscillatorNetworkState,
    AmphibiousData,
)
from farms_amphibious.network.network import NetworkODE
from farms_amphibious.experiment.simulation import profile
from farms_amphibious.utils.network import plot_networks_maps
import farms_pylog as pylog


def animat_options():
    """Animat options"""
    # Options
    morphology = AmphibiousMorphologyOptions.from_options({
        'n_joints_body': 11,
        'n_legs': 4,
        'n_dof_legs': 4,
    })
    control = AmphibiousControlOptions.from_options({
        'kinematics_file': '',
        'drives': AmphibiousDrives.from_options({
            'drive_forward': 2,
            'drive_turn': 0,
            'drive_left': 0,
            'drive_right': 0,
        }),
        'network': AmphibiousNetworkOptions.from_options({
            'oscillators': AmphibiousOscillatorOptions.from_options(
                {
                    'body_head_amplitude': 0,
                    'body_tail_amplitude': 0,
                    'body_stand_amplitude': 0.2,
                    'legs_amplitude': [0.8, np.pi/32, np.pi/4, np.pi/8],
                    'body_stand_shift': np.pi/4,
                }
            ),
            'connectivity': AmphibiousConnectivityOptions.from_options(
                {
                    'body_phase_bias': 2*np.pi/morphology.n_joints_body,
                    'leg_phase_follow': np.pi,
                    'w_legs2body': 3e1,
                    'w_sens_contact_i': -2e0,
                    'w_sens_contact_e': 2e0,  # +3e-1
                    'w_sens_hyfro_freq': -1,
                    'w_sens_hydro_amp': 1,
                }
            ),
            'joints': AmphibiousJointsOptions.from_options(
                {
                    'legs_offsets_walking': [0, np.pi/32, 0, np.pi/8],
                    'legs_offsets_swimming': [-2*np.pi/5, 0, 0, 0],
                    'gain_amplitude': [1 for _ in range(morphology.n_joints())],
                    'gain_offset': [1 for _ in range(morphology.n_joints())],
                }
            ),
            'sensors': None
        })
    })
    control.network.update(
        morphology.n_joints_body,
        morphology.n_dof_legs,
    )
    return morphology, control


def run_simulation(network, n_iterations, timestep):
    """Run simulation"""
    for iteration in range(n_iterations-1):
        network.control_step(iteration, iteration*timestep, timestep)


def simulation(times, morphology, control):
    """Simulation"""
    timestep = times[1] - times[0]
    n_iterations = len(times)

    # Animat data
    animat_data = AmphibiousData.from_options(
        AmphibiousOscillatorNetworkState.default_state(
            n_iterations,
            morphology,
        ),
        morphology,
        control,
        n_iterations
    )

    # Animat network
    network = NetworkODE(animat_data)
    profile(
        run_simulation,
        network=network,
        n_iterations=n_iterations,
        timestep=timestep,
    )

    return network, animat_data


def analysis(data, times, morphology):
    """Analysis"""
    # Network information
    sep = '\n  - '
    pylog.info(
        'Oscillator connectivity information'
        + sep.join([
            'O_{} <- O_{} (w={}, theta={})'.format(
                int(connection[0]+0.5),
                int(connection[1]+0.5),
                connection[2],
                connection[3],
            )
            for connection in data.network.connectivity.array
        ])
    )
    pylog.info(
        'Contacts connectivity information'
        + sep.join([
            'O_{} <- contact_{} (frequency_gain={})'.format(
                int(connection[0]+0.5),
                int(connection[1]+0.5),
                connection[2],
            )
            for connection in data.network.contacts_connectivity.array
        ])
    )
    pylog.info(
        'Hydrodynamics connectivity information'
        + sep.join([
            'O_{} <- link_{} (phase_gain={}, amplitude_gain={})'.format(
                int(connection[0]+0.5),
                int(connection[1]+0.5),
                connection[2],
                connection[3],
            )
            for connection in data.network.hydro_connectivity.array
        ])
    )
    sep = '\n'
    pylog.info(
        (sep.join([
            'Network infromation:',
            '  - Oscillators shape: {}',
            '  - Connectivity shape: {}',
            '  - Contacts connectivity shape: {}',
            '  - Hydro connectivity shape: {}',
        ])).format(
            np.shape(data.network.oscillators.array),
            np.shape(data.network.connectivity.array),
            np.shape(data.network.contacts_connectivity.array),
            np.shape(data.network.hydro_connectivity.array),
        )
    )

    # Plot data
    # data.plot(times)
    data.state.plot_phases(times)
    data.state.plot_amplitudes(times)

    # Network
    plot_networks_maps(morphology, data)


def main(filename='cpg_network.h5'):
    """Main"""
    times = np.arange(0, 10, 1e-3)
    morphology, control = animat_options()
    _, animat_data = simulation(times, morphology, control)

    # Save data
    pylog.debug('Saving data to {}'.format(filename))
    animat_data.to_file(filename)
    pylog.debug('Save complete')

    # Save options
    morphology_filename = 'options_morphology.yaml'
    control_filename = 'options_control.yaml'
    morphology.save(morphology_filename)
    control.save(control_filename)

    # Load options
    morphology = AmphibiousMorphologyOptions.load(morphology_filename)
    control = AmphibiousControlOptions.load(control_filename)

    # Load from file
    pylog.debug('Loading data from {}'.format(filename))
    data = AmphibiousData.from_file(
        filename,
        2*morphology.n_joints()
    )
    pylog.debug('Load complete')

    # # Post-processing
    # analysis(data, times, morphology)

    # Show
    plt.show()


if __name__ == '__main__':
    main()
