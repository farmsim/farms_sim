"""CPG network"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import colorConverter

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
from farms_amphibious.model.convention import AmphibiousConvention
from farms_amphibious.network.network import NetworkODE
from farms_amphibious.experiment.simulation import profile
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


def rotate(vector, theta):
    """Rotate vector"""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array(((cos_t, -sin_t), (sin_t, cos_t)))
    return np.dot(rotation, vector)


def direction(vector1, vector2):
    """Unit direction"""
    return (vector2-vector1)/np.linalg.norm(vector2-vector1)


def connect_positions(source, destination, dir_shift, perp_shift):
    """Connect positions"""
    connection_direction = direction(source, destination)
    connection_perp = rotate(connection_direction, 0.5*np.pi)
    new_source = (
        source
        + dir_shift*connection_direction
        + perp_shift*connection_perp
    )
    new_destination = (
        destination
        - dir_shift*connection_direction
        + perp_shift*connection_perp
    )
    return new_source, new_destination


def draw_nodes(positions, radius, color, prefix):
    """Draw nodes"""
    nodes = [
        plt.Circle(
            position,
            radius,
            facecolor=color,
            edgecolor=0.7*np.array(colorConverter.to_rgb(color)),
            linewidth=2,
        )  # fill=False, clip_on=False
        for position in positions
    ]

    nodes_texts = [
        plt.text(
            # position[0]+radius, position[1]+radius,
            position[0], position[1],
            '{}{}'.format(prefix, i),
            # transform=axes.transAxes,
            # va="bottom",
            # ha="left",
            va="center",
            ha="center",
            fontsize=8,
            color='k',
        )
        for i, position in enumerate(positions)
    ]

    return nodes, nodes_texts


def draw_connectivity(sources, destinations, radius, connectivity, rad, color):
    """Draw nodes"""
    node_connectivity = [
        patches.FancyArrowPatch(
            *connect_positions(
                source,
                destination,
                radius,
                -radius/2 if rad > 0 else 0
            ),
            arrowstyle=patches.ArrowStyle(
                stylename='Simple',
                head_length=8,
                head_width=4,
                tail_width=0.5,
            ),
            connectionstyle=patches.ConnectionStyle(
                'Arc3',
                rad=rad,
            ),
            color=color,
        )
        for source, destination in np.array([
            [
                sources[int(connection[1]+0.5)],
                destinations[int(connection[0]+0.5)],
            ]
            for connection in connectivity
        ])
    ]
    return node_connectivity


def draw_network(source, destination, radius, connectivity, **kwargs):
    """Draw network"""
    # Arguments
    prefix = kwargs.pop('prefix')
    rad = kwargs.pop('rad')
    color = kwargs.pop('color')
    alpha = kwargs.pop('alpha')

    # Nodes
    nodes, nodes_texts = draw_nodes(
        positions=source,
        radius=radius,
        color=color,
        prefix=prefix,
    )
    node_connectivity = draw_connectivity(
        sources=source,
        destinations=destination,
        radius=radius,
        connectivity=connectivity,
        rad=rad,
        color=colorConverter.to_rgb(color)+(alpha,),
    )
    return nodes, nodes_texts, node_connectivity


def plot_network(n_oscillators, data, **kwargs):
    """Plot network"""
    offset = kwargs.pop('offset', 1)
    radius = kwargs.pop('radius', 0.3)
    margin_x = kwargs.pop('margin_x', 2)
    margin_y = kwargs.pop('margin_y', 7)
    alpha = kwargs.pop('alpha', 0.3)
    title = kwargs.pop('title', 'Network')
    rads = kwargs.pop('rads', [0.2, 0.0, 0.0])

    plt.figure(num=title, figsize=(12, 10))

    axes = plt.gca()
    axes.cla()
    plt.title(title)
    axes.set_xlim((-margin_x, n_oscillators-1+margin_x))
    axes.set_ylim((-offset-margin_y, offset+margin_y))
    axes.set_aspect('equal', adjustable='box')
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    plt.tight_layout()

    # Oscillators
    oscillator_positions = np.array(
        [
            [2*osc_x, side_y]
            for osc_x in range(n_oscillators//2)
            for side_y in [-offset, offset]
        ] + [
            [leg_x+osc_side_x+joint_y, joint_y*side_x]
            for leg_x in [1, 11]
            for side_x in [-1, 1]
            for joint_y in [3, 4, 5, 6]
            for osc_side_x in [-1, 1]
        ]
    )
    osc_conn_cond = kwargs.pop(
        'osc_conn_cond',
        lambda osc0, osc1: True
    )
    oscillators, oscillators_texts, oscillator_connectivity = draw_network(
        source=oscillator_positions,
        destination=oscillator_positions,
        radius=radius,
        connectivity=[
            connection
            for connection in data.network.connectivity.array
            if osc_conn_cond(connection[0], connection[1])
        ],
        prefix='O_',
        rad=rads[0],
        color='C2',
        alpha=alpha,
    )

    # Contacts
    contacts_positions = np.array([
        [leg_x, side_y]
        for leg_x in [1+6, 11+6]
        for side_y in [-7, 7]
    ])
    contact_sensors, contact_sensor_texts, contact_connectivity = draw_network(
        source=contacts_positions,
        destination=oscillator_positions,
        radius=radius,
        connectivity=data.network.contacts_connectivity.array,
        prefix='C_',
        rad=rads[1],
        color='C1',
        alpha=alpha,
    )

    # Hydrodynamics
    hydrodynamics_positions = np.array([
        [2*osc_x+1, 0]
        for osc_x in range(-1, n_oscillators//2)
    ])
    hydro_sensors, hydro_sensor_texts, hydro_connectivity = draw_network(
        source=hydrodynamics_positions,
        destination=oscillator_positions,
        radius=radius,
        connectivity=data.network.hydro_connectivity.array,
        prefix='H_',
        rad=rads[2],
        color='C0',
        alpha=2*alpha
    )

    # Show elements
    [
        show_oscillators,
        show_contacts,
        show_hydrodynamics,
        show_oscillator_connectivity,
        show_contacts_connectivity,
        show_hydrodynamics_connectivity,
    ] = [
        kwargs.pop(key, True)
        for key in [
                'show_oscillators',
                'show_contacts',
                'show_hydrodynamics',
                'show_oscillator_connectivity',
                'show_contacts_connectivity',
                'show_hydrodynamics_connectivity',
        ]
    ]
    if show_oscillator_connectivity:
        for arrow in oscillator_connectivity:
            axes.add_artist(arrow)
    if show_contacts_connectivity:
        for arrow in contact_connectivity:
            axes.add_artist(arrow)
    if show_hydrodynamics_connectivity:
        for arrow in hydro_connectivity:
            axes.add_artist(arrow)
    if show_oscillators:
        for circle, text in zip(oscillators, oscillators_texts):
            axes.add_artist(circle)
            axes.add_artist(text)
    if show_contacts:
        for circle, text in zip(contact_sensors, contact_sensor_texts):
            axes.add_artist(circle)
            axes.add_artist(text)
    if show_hydrodynamics:
        for circle, text in zip(hydro_sensors, hydro_sensor_texts):
            axes.add_artist(circle)
            axes.add_artist(text)


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

    # Plot tools
    convention = AmphibiousConvention(**morphology)
    info = convention.oscindex2information
    body2body = lambda osc0, osc1: info(osc0)['body'] and info(osc1)['body']
    leg2body = lambda osc0, osc1: info(osc0)['body'] and not info(osc1)['body']
    body2leg = lambda osc0, osc1: not info(osc0)['body'] and info(osc1)['body']
    leg2leg = (
        lambda osc0, osc1: not info(osc0)['body'] and not info(osc1)['body']
    )
    leg2sameleg = (
        lambda osc0, osc1: (
            not info(osc0)['body']
            and not info(osc1)['body']
            and info(osc0)['leg'] == info(osc1)['leg']
        )
    )
    leg2diffleg = (
        lambda osc0, osc1: (
            not info(osc0)['body']
            and not info(osc1)['body']
            and info(osc0)['leg'] != info(osc1)['leg']
        )
    )

    # Plot network
    # plot_network(
    #     n_oscillators=2*morphology.n_joints_body,
    #     data=data,
    #     title='Complete CPG Network',
    # )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators complete connectivity',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators body2body connectivity',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        osc_conn_cond=body2body,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators body2limb connectivity',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        osc_conn_cond=body2leg,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators limb2body connectivity',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        osc_conn_cond=leg2body,
        rads=[0.05, 0.0, 0.0],
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators limb2limb connectivity',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        osc_conn_cond=leg2leg,
        rads=[0.05, 0.0, 0.0],
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators intralimb connectivity',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        osc_conn_cond=leg2sameleg,
        rads=[0.2, 0.0, 0.0],
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators interlimb connectivity',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        osc_conn_cond=leg2diffleg,
        rads=[0.05, 0.0, 0.0],
    )
    # plot_network(
    #     n_oscillators=2*morphology.n_joints_body,
    #     data=data,
    #     title='Contacts',
    #     show_oscillator_connectivity=False,
    #     show_hydrodynamics_connectivity=False,
    # )
    # plot_network(
    #     n_oscillators=2*morphology.n_joints_body,
    #     data=data,
    #     title='Hydrodynamics',
    #     show_oscillator_connectivity=False,
    #     show_contacts_connectivity=False,
    # )


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

    # Post-processing
    analysis(data, times, morphology)

    # demo()
    plt.show()


if __name__ == '__main__':
    main()
