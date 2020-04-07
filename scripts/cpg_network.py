"""CPG network"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
import farms_pylog as pylog


def demo_con_style(ax, connectionstyle):
    x1, y1 = 0.3, 0.2
    x2, y2 = 0.8, 0.6

    ax.plot([x1, x2], [y1, y2], ".")
    ax.annotate(
        "",
        xy=(x1, y1), xycoords='data',
        xytext=(x2, y2), textcoords='data',
        arrowprops=dict(
            arrowstyle="->", color="0.5",
            shrinkA=5, shrinkB=5,
            patchA=None, patchB=None,
            connectionstyle=connectionstyle,
        ),
    )

    ax.text(
        .05, .95, connectionstyle.replace(",", ",\n"),
        transform=ax.transAxes, ha="left", va="top"
    )


def demo():
    fig, axs = plt.subplots(3, 5, figsize=(8, 4.8))
    demo_con_style(axs[0, 0], "angle3,angleA=90,angleB=0")
    demo_con_style(axs[1, 0], "angle3,angleA=0,angleB=90")
    demo_con_style(axs[0, 1], "arc3,rad=0.")
    demo_con_style(axs[1, 1], "arc3,rad=0.3")
    demo_con_style(axs[2, 1], "arc3,rad=-0.3")
    demo_con_style(axs[0, 2], "angle,angleA=-90,angleB=180,rad=0")
    demo_con_style(axs[1, 2], "angle,angleA=-90,angleB=180,rad=5")
    demo_con_style(axs[2, 2], "angle,angleA=-90,angleB=10,rad=5")
    demo_con_style(axs[0, 3], "arc,angleA=-90,angleB=0,armA=30,armB=30,rad=0")
    demo_con_style(axs[1, 3], "arc,angleA=-90,angleB=0,armA=30,armB=30,rad=5")
    demo_con_style(axs[2, 3], "arc,angleA=-90,angleB=0,armA=0,armB=40,rad=0")
    demo_con_style(axs[0, 4], "bar,fraction=0.3")
    demo_con_style(axs[1, 4], "bar,fraction=-0.3")
    demo_con_style(axs[2, 4], "bar,angle=180,fraction=-0.2")

    for ax in axs.flat:
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[], aspect=1)
    fig.tight_layout(pad=0.2)

    plt.show()


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
    for iteration in range(n_iterations-1):
        network.control_step(iteration, iteration*timestep, timestep)

    return network, animat_data


def analysis(data, times, morphology):
    """Analysis"""
    # Network information
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

    # # Plot data
    # data.plot(times)

    # Plot network
    n_oscillators = 2*morphology.n_joints_body
    offset = 0.5
    radius = 0.2
    margin_x = 1
    margin_y = 1
    plt.figure('CPGnetwork', figsize=(12, 3))
    axes = plt.gca()
    axes.cla()
    axes.set_xlim((-margin_x, n_oscillators/2-1+margin_x))
    axes.set_ylim((-offset-margin_y, offset+margin_y))
    axes.set_aspect('equal', adjustable='box')

    positions = [
        [osc_i, side]
        for osc_i in range(n_oscillators//2)
        for side in [-offset, offset]
    ]

    circles = [
        plt.Circle(position, radius, color='g')  # fill=False, clip_on=False
        for position in positions
    ]

    texts = [
        axes.text(
            position[0]+radius, position[1]+radius,
            'O_{}'.format(i),
            # transform=axes.transAxes,
            va="bottom",
            ha="left",
        )
        for i, position in enumerate(positions)
    ]

    offx = np.array([radius/2, 0])
    offy = np.array([0, radius])
    arrows = [
        patches.FancyArrowPatch(
            position+offx+offy,
            position2+offx-offy,
            arrowstyle=patches.ArrowStyle(  # 'Simple,tail_width=0.5,head_width=4,head_length=8',
                stylename='Simple',
                head_length=8,
                head_width=4,
                tail_width=0.5,
            ),
            connectionstyle=patches.ConnectionStyle(
                'Arc3',
                rad=0.3
            ),
            color='k',
        )
        for osc_i, (position, position2) in enumerate(
            zip(positions[0::2], positions[1::2])
        )
    ] + [
        patches.FancyArrowPatch(
            position-offx-offy,
            position2-offx+offy,
            arrowstyle=patches.ArrowStyle(  # 'Simple,tail_width=0.5,head_width=4,head_length=8',
                stylename='Simple',
                head_length=8,
                head_width=4,
                tail_width=0.5,
            ),
            connectionstyle=patches.ConnectionStyle(
                'Arc3',
                rad=0.3
            ),
            color='k',
        )
        for osc_i, (position, position2) in enumerate(
            zip(positions[1::2], positions[0::2])
        )
    ]
    for circle, text, arrow in zip(circles, texts, arrows):
        axes.add_artist(circle)
        axes.add_artist(text)
        axes.add_artist(arrow)


def main(filename='cpg_network.h5'):
    """Main"""
    times = np.arange(0, 1, 1e-3)
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
    data = AmphibiousData.from_file(filename)
    pylog.debug('Load complete')

    # Post-processing
    analysis(data, times, morphology)

    # demo()
    plt.show()


if __name__ == '__main__':
    main()
