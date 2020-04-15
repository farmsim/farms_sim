"""Network"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.colors import colorConverter, Normalize, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import farms_pylog as pylog
from ..model.convention import AmphibiousConvention


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


def draw_connectivity(sources, destinations, connectivity, **kwargs):
    """Draw nodes"""
    radius = kwargs.pop('radius')
    rad = kwargs.pop('rad')
    color = kwargs.pop('color')
    alpha = kwargs.pop('alpha')
    if isinstance(color, ListedColormap):
        color_values = kwargs.pop('color_values')
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
            color=colorConverter.to_rgb(
                color(color_values[connection_i])
                if isinstance(color, ListedColormap)
                else color
            )+(alpha,),
        )
        for connection_i, (source, destination) in enumerate([
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
    color_nodes = kwargs.pop('color_nodes')
    color_arrows = kwargs.pop('color_arrows', None)
    options = {}
    alpha = kwargs.pop('alpha')
    if color_arrows is None:
        color_arrows = colorConverter.to_rgb(color_nodes)
    else:
        weights = kwargs.pop('weights')
        if list(weights):
            weight_min = np.min(weights)
            weight_max = np.max(weights)
            if weight_max == weight_min:
                weight_max += 1e-3
                weight_min -= 1e-3
            options['color_values'] = (
                (weights-weight_min)/(weight_max-weight_min)
            )
        else:
            options['color_values'] = None

    # Nodes
    nodes, nodes_texts = draw_nodes(
        positions=source,
        radius=radius,
        color=color_nodes,
        prefix=prefix,
    )
    node_connectivity = draw_connectivity(
        sources=source,
        destinations=destination,
        radius=radius,
        connectivity=connectivity,
        rad=rad,
        color=color_arrows,
        alpha=alpha,
        **options
    )
    return nodes, nodes_texts, node_connectivity


def plot_network(n_oscillators, data, **kwargs):
    """Plot network"""
    # Options
    offset = kwargs.pop('offset', 1)
    radius = kwargs.pop('radius', 0.3)
    margin_x = kwargs.pop('margin_x', 2)
    margin_y = kwargs.pop('margin_y', 7)
    alpha = kwargs.pop('alpha', 0.3)
    title = kwargs.pop('title', 'Network')
    rads = kwargs.pop('rads', [0.2, 0.0, 0.0])
    use_colorbar = kwargs.pop('use_colorbar', False)

    # Create figure
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

    # Colorbar
    if use_colorbar:
        cmap = plt.get_cmap(kwargs.pop('cmap', 'viridis'))

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
    connections = np.array([
        [connection[0], connection[1], weight, phase]
        for connection, weight, phase in zip(
            data.network.osc_connectivity.connections.array,
            data.network.osc_connectivity.weights.array,
            data.network.osc_connectivity.desired_phases.array,
        )
        if osc_conn_cond(connection[0], connection[1])
    ])
    options = {}
    use_weights = use_colorbar and kwargs.pop('oscillator_weights', False)
    if use_weights:
        if connections.any():
            options['weights'] = connections[:, 2]
            vmin = np.min(connections[:, 2])
            vmax = np.max(connections[:, 2])
        else:
            options['weights'] = []
            vmin, vmax = 0, 1
    options = {}
    use_weights = use_colorbar and kwargs.pop('oscillator_phases', False)
    if use_weights:
        if connections.any():
            options['weights'] = connections[:, 3]
            vmin = np.min(connections[:, 3])
            vmax = np.max(connections[:, 3])
        else:
            options['weights'] = []
            vmin, vmax = 0, 1
    oscillators, oscillators_texts, oscillator_connectivity = draw_network(
        source=oscillator_positions,
        destination=oscillator_positions,
        radius=radius,
        connectivity=connections,
        prefix='O_',
        rad=rads[0],
        color_nodes='C2',
        color_arrows=cmap if use_weights else None,
        alpha=alpha,
        **options,
    )

    # Contacts
    contacts_positions = np.array([
        [leg_x, side_y]
        for leg_x in [1+6, 11+6]
        for side_y in [-7, 7]
    ])
    contact_conn_cond = kwargs.pop(
        'contact_conn_cond',
        lambda osc0, osc1: True
    )
    connections = np.array([
        [connection[0], connection[1], weight]
        for connection, weight in zip(
            data.network.contacts_connectivity.connections.array,
            data.network.contacts_connectivity.weights.array,
        )
        if contact_conn_cond(connection[0], connection[1])
    ])
    options = {}
    use_weights = use_colorbar and kwargs.pop('contacts_weights', False)
    if use_weights:
        if connections.any():
            options['weights'] = connections[:, 2]
            vmin = np.min(connections[:, 2])
            vmax = np.max(connections[:, 2])
        else:
            options['weights'] = []
            vmin, vmax = 0, 1
    contact_sensors, contact_sensor_texts, contact_connectivity = draw_network(
        source=contacts_positions,
        destination=oscillator_positions,
        radius=radius,
        connectivity=connections,
        prefix='C_',
        rad=rads[1],
        color_nodes='C1',
        color_arrows=cmap if use_weights else None,
        alpha=alpha,
        **options,
    )

    # Hydrodynamics
    hydrodynamics_positions = np.array([
        [2*osc_x+1, 0]
        for osc_x in range(-1, n_oscillators//2)
    ])
    hydro_conn_cond = kwargs.pop(
        'hydro_conn_cond',
        lambda osc0, osc1: True
    )
    connections = np.array([
        [connection[0], connection[1], frequency, amplitude]
        for connection, frequency, amplitude in zip(
            data.network.hydro_connectivity.connections.array,
            data.network.hydro_connectivity.frequency.array,
            data.network.hydro_connectivity.amplitude.array,
        )
        if hydro_conn_cond(connection[0], connection[1])
    ])
    options = {}
    use_weights = use_colorbar and kwargs.pop('hydro_frequency_weights', False)
    if use_weights:
        if connections.any():
            options['weights'] = connections[:, 2]
            vmin = np.min(connections[:, 2])
            vmax = np.max(connections[:, 2])
        else:
            options['weights'] = []
            vmin, vmax = 0, 1
    use_weights = use_colorbar and kwargs.pop('hydro_amplitude_weights', False)
    if use_weights:
        if connections.any():
            options['weights'] = connections[:, 3]
            vmin = np.min(connections[:, 3])
            vmax = np.max(connections[:, 3])
        else:
            options['weights'] = []
            vmin, vmax = 0, 1
    hydro_sensors, hydro_sensor_texts, hydro_connectivity = draw_network(
        source=hydrodynamics_positions,
        destination=oscillator_positions,
        radius=radius,
        connectivity=[
            connection
            for connection in data.network.hydro_connectivity.connections.array
            if hydro_conn_cond(connection[0], connection[1])
        ],
        prefix='H_',
        rad=rads[2],
        color_nodes='C0',
        color_arrows=cmap if use_weights else None,
        alpha=2*alpha,
        **options,
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
    if use_colorbar:
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        pylog.debug('{}: {}, {}'.format(title, vmin, vmax))
        plt.colorbar(
            mappable=cm.ScalarMappable(
                norm=Normalize(
                    vmin=vmin,
                    vmax=vmax,
                    clip=True,
                ),
                cmap=cmap,
            ),
            cax=cax,
        )
        plt.sca(axes)


def plot_networks_maps(morphology, data):
    """Plot network maps"""
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
    contact2sameleg = (
        lambda osc0, osc1: (
            not info(osc0)['body']
            and info(osc0)['leg'] == osc1
            # and osc1 in (1, 2)
        )
    )
    contact2diffleg = (
        lambda osc0, osc1: (
            not info(osc0)['body']
            and info(osc0)['leg'] != osc1
            # and osc1 in (1, 2)
        )
    )

    # Plot network
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Complete network',
    )

    # Plot network oscillator weights connectivity
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators complete connectivity',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        use_colorbar=True,
        oscillator_weights=True,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators body2body connectivity',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        osc_conn_cond=body2body,
        use_colorbar=True,
        oscillator_weights=True,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators body2limb connectivity',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        osc_conn_cond=body2leg,
        use_colorbar=True,
        oscillator_weights=True,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators limb2body connectivity',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        osc_conn_cond=leg2body,
        rads=[0.05, 0.0, 0.0],
        use_colorbar=True,
        oscillator_weights=True,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators limb2limb connectivity',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        osc_conn_cond=leg2leg,
        rads=[0.05, 0.0, 0.0],
        use_colorbar=True,
        oscillator_weights=True,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators intralimb connectivity',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        osc_conn_cond=leg2sameleg,
        rads=[0.2, 0.0, 0.0],
        use_colorbar=True,
        oscillator_weights=True,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators interlimb connectivity',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        osc_conn_cond=leg2diffleg,
        rads=[0.05, 0.0, 0.0],
        use_colorbar=True,
        oscillator_weights=True,
    )

    # Plot network oscillator phases connectivity
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators complete phases',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        use_colorbar=True,
        oscillator_phases=True,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators body2body phases',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        osc_conn_cond=body2body,
        use_colorbar=True,
        oscillator_phases=True,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators body2limb phases',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        osc_conn_cond=body2leg,
        use_colorbar=True,
        oscillator_phases=True,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators limb2body phases',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        osc_conn_cond=leg2body,
        rads=[0.05, 0.0, 0.0],
        use_colorbar=True,
        oscillator_phases=True,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators limb2limb phases',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        osc_conn_cond=leg2leg,
        rads=[0.05, 0.0, 0.0],
        use_colorbar=True,
        oscillator_phases=True,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators intralimb phases',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        osc_conn_cond=leg2sameleg,
        rads=[0.2, 0.0, 0.0],
        use_colorbar=True,
        oscillator_phases=True,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Oscillators interlimb phases',
        show_contacts_connectivity=False,
        show_hydrodynamics_connectivity=False,
        osc_conn_cond=leg2diffleg,
        rads=[0.05, 0.0, 0.0],
        use_colorbar=True,
        oscillator_phases=True,
    )

    # Plot contacts connectivity
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Contacts complete connectivity',
        show_oscillator_connectivity=False,
        show_hydrodynamics_connectivity=False,
        use_colorbar=True,
        contacts_weights=True,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Contacts intralimb connectivity',
        show_oscillator_connectivity=False,
        show_hydrodynamics_connectivity=False,
        contact_conn_cond=contact2sameleg,
        use_colorbar=True,
        contacts_weights=True,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Contacts interlimb connectivity',
        show_oscillator_connectivity=False,
        show_hydrodynamics_connectivity=False,
        contact_conn_cond=contact2diffleg,
        use_colorbar=True,
        contacts_weights=True,
    )

    # Plot hydrodynamics connectivity
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Hydrodynamics complete connectivity',
        show_oscillator_connectivity=False,
        show_contacts_connectivity=False,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Hydrodynamics frequency connectivity',
        show_oscillator_connectivity=False,
        show_contacts_connectivity=False,
        use_colorbar=True,
        hydro_frequency_weights=True,
    )
    plot_network(
        n_oscillators=2*morphology.n_joints_body,
        data=data,
        title='Hydrodynamics amplitude connectivity',
        show_oscillator_connectivity=False,
        show_contacts_connectivity=False,
        use_colorbar=True,
        hydro_amplitude_weights=True,
    )
