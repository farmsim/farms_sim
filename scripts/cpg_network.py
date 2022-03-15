"""Network"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import MT19937, RandomState, SeedSequence

import farms_pylog as pylog
from farms_data.utils.profile import profile
from farms_data.amphibious.data import AmphibiousData
from farms_amphibious.model.convention import AmphibiousConvention
from farms_amphibious.control.network import NetworkODE
from farms_amphibious.utils.network import plot_networks_maps
from farms_amphibious.utils.prompt import prompt
from farms_amphibious.model.options import (
    AmphibiousMorphologyOptions,
    AmphibiousControlOptions,
)


def animat_options():
    """Animat options"""
    # Options
    morphology = AmphibiousMorphologyOptions.from_options({
        'n_joints_body': 11,
        'n_legs': 4,
        'n_dof_legs': 4,
    })
    options = {
        'kinematics_file': '',
        'drive_forward': 2,
        'drive_turn': 0,
        'drive_left': 0,
        'drive_right': 0,
        'body_head_amplitude': 0,
        'body_tail_amplitude': 0,
        'body_stand_amplitude': 0.2,
        'legs_amplitude': [np.pi/4, np.pi/32, np.pi/4, np.pi/8],
        'body_stand_shift': np.pi/4,
        'body_phase_bias': 2*np.pi/morphology.n_joints_body,
        'leg_phase_follow': np.pi,
        'w_legs2body': 3e1,
        'weight_sens_contact_intralimb': -0.5,
        'weight_sens_contact_opposite': 2,
        'weight_sens_contact_following': 0,
        'weight_sens_contact_diagonal': 0,
        'weight_sens_hydro_freq': -1,
        'weight_sens_hydro_amp': 1,
        'legs_offsets_walking': [0, np.pi/32, 0, np.pi/8],
        'legs_offsets_swimming': [-2*np.pi/5, 0, 0, 0],
    }
    control = AmphibiousControlOptions.from_options(options)
    convention = AmphibiousConvention.from_morphology(morphology)
    control.defaults_from_convention(convention, options)
    return morphology, control, convention


def run_simulation(network, n_iterations, timestep):
    """Run simulation"""
    for iteration in range(n_iterations-1):
        network.step(iteration, iteration*timestep, timestep)


def simulation(times, control, convention):
    """Simulation"""
    timestep = times[1] - times[0]
    n_iterations = len(times)

    # Animat data
    control.network.drives[0].initial_value = 2
    control.network.drives[1].initial_value = 0
    random_state = RandomState(MT19937(SeedSequence(123456789)))
    animat_data = AmphibiousData.from_options(
        control=control,
        initial_state=1e-1*random_state.rand(convention.n_states()),
        n_iterations=n_iterations,
        timestep=timestep,
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
    # Plot data
    if prompt('Show plots', False):
        # data.plot(times)
        data.state.plot_phases(times)
        data.state.plot_amplitudes(times)

    # Network
    if prompt('Show connectivity maps', False):
        sep = '\n  - '
        pylog.info(
            'Oscillator connectivity information\n%s',
            sep.join([
                f'O_{connection[0]} <- O_{connection[1]}'
                f' (w={weight}, theta={phase})'
                for connection, weight, phase in zip(
                    data.network.osc_connectivity.connections.array,
                    data.network.osc_connectivity.weights.array,
                    data.network.osc_connectivity.desired_phases.array,
                )
            ])
        )
        pylog.info(
            'Contacts connectivity information\n%s',
            sep.join([
                f'O_{connection[0]} <- contact_{connection[1]}'
                f' (frequency_gain={weight})'
                for connection, weight in zip(
                    data.network.contacts_connectivity.connections.array,
                    data.network.contacts_connectivity.weights.array,
                )
            ])
        )
        pylog.info(
            'Hydrodynamics connectivity information\n%s',
            sep.join([
                f'O_{connection[0]} <- link_{connection[1]}'
                f' (type={connection[2]}, weight={weight})'
                for connection, weight in zip(
                    data.network.hydro_connectivity.connections.array,
                    data.network.hydro_connectivity.weights.array,
                )
            ])
        )
        sep = '\n'
        pylog.info(
            sep.join([
                'Network infromation:',
                '  - Oscillators:',
                '     - Intrinsic frequencies: {}',
                '     - Nominal amplitudes: {}',
                '     - Rates: {}',
                '  - Connectivity shape: {}',
                '  - Contacts connectivity shape: {}',
                '  - Hydro connectivity shape: {}',
            ]),
            np.shape(data.network.oscillators.intrinsic_frequencies.array),
            np.shape(data.network.oscillators.nominal_amplitudes.array),
            np.shape(data.network.oscillators.rates.array),
            np.shape(data.network.osc_connectivity.connections.array),
            np.shape(data.network.contacts_connectivity.connections.array),
            np.shape(data.network.hydro_connectivity.connections.array),
        )

        plot_networks_maps(morphology, data)


def main(filename='cpg_network.h5'):
    """Main"""
    times = np.arange(0, 10, 1e-3)
    morphology, control, convention = animat_options()
    _, animat_data = simulation(times, control, convention)

    # Save data
    pylog.debug('Saving data to %s', filename)
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
    pylog.debug('Loading data from %s', filename)
    data = AmphibiousData.from_file(filename)
    pylog.debug('Load complete')

    # Post-processing
    analysis(data, times, morphology)

    # Show
    plt.show()


if __name__ == '__main__':
    main()
