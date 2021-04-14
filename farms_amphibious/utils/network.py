"""Network"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import colorConverter, Normalize, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import farms_pylog as pylog
from farms_data.amphibious.animat_data_cy import ConnectionType

from ..model.convention import AmphibiousConvention
from ..control.network import NetworkODE


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
            # va='bottom',
            # ha='left',
            va='center',
            ha='center',
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


def create_colorbar(axes, cmap, vmin, vmax, size='5%', pad=0.05, **kwargs):
    """Colorbar"""
    return plt.colorbar(
        mappable=cm.ScalarMappable(
            norm=Normalize(
                vmin=vmin,
                vmax=vmax,
                clip=True,
            ),
            cmap=cmap,
        ),
        **kwargs,
    )


class NetworkFigure:
    """Network figure"""

    def __init__(self, morphology, data):
        super(NetworkFigure, self).__init__()
        self.data = data
        self.morphology = morphology

        # Plot
        self.axes = None
        self.figure = None

        # Artists
        self.oscillators = None
        self.contact_sensors = None
        self.hydro_sensors = None

        # Animation
        self.animation = None
        self.timestep = self.data.timestep
        self.time = None
        self.n_iterations = np.shape(self.data.sensors.links.array)[0]
        self.interval = 25
        self.n_frames = round(self.n_iterations*self.timestep / (1e-3*self.interval))
        self.network = NetworkODE(self.data)
        self.cmap_phases = plt.get_cmap('Greens')
        self.cmap_contacts = plt.get_cmap('Oranges')
        self.cmap_contact_max = 2e-1
        self.cmap_hydro = plt.get_cmap('Blues')
        self.cmap_hydro_max = 2e-2

    def animate(self):
        """Setup animation"""
        plt.figure(num=self.figure.number)
        self.time = plt.text(
            x=-1, y=-7,
            s='Time: {} [s]'.format(0),
            va='center',
            ha='left',
            fontsize=16,
            color='k',
        )
        # Oscillators
        divider = make_axes_locatable(self.axes)
        size = '5%'
        pad = 0.5  # 0.05
        cax = divider.append_axes('right', size=size, pad=pad)
        cbar = create_colorbar(
            axes=self.axes,
            cmap=self.cmap_phases,
            vmin=0, vmax=1,
            cax=cax,
        )
        cbar.ax.set_ylabel('Oscillators output', rotation=270)
        cbar.ax.get_yaxis().labelpad = -30

        # Contacts
        cax = divider.append_axes('right', size=size, pad=pad)
        cbar = create_colorbar(
            self.axes,
            cmap=self.cmap_contacts,
            vmin=0, vmax=self.cmap_contact_max,
            cax=cax,
        )
        cbar.ax.set_ylabel('Contacts forces [N]', rotation=270)
        cbar.ax.get_yaxis().labelpad = -45

        # Hydroynamics
        cax = divider.append_axes('right', size=size, pad=pad)
        cbar = create_colorbar(
            axes=self.axes,
            cmap=self.cmap_hydro,
            vmin=0, vmax=self.cmap_hydro_max,
            cax=cax,
        )
        cbar.ax.set_ylabel('Hydrodynamics forces [N]', rotation=270)
        cbar.ax.get_yaxis().labelpad = -50

        # Animation
        self.animation = FuncAnimation(
            fig=self.figure,
            func=self.animation_update,
            frames=np.arange(self.n_frames),
            init_func=self.animation_init,
            blit=False,
            interval=self.interval,
            cache_frame_data=False,
        )

    def animation_elements(self):
        """Animation elements"""
        return [self.time] + (
            self.oscillators + self.contact_sensors + self.hydro_sensors
        )

    def animation_init(self):
        """Animation  init"""
        return self.animation_elements()

    def animation_update(self, frame):
        """Animation update"""
        # Time
        iteration = np.rint(frame/self.n_frames*self.n_iterations).astype(int)
        self.time.set_text('Time: {:02.1f} [s]'.format(frame*1e-3*self.interval))

        # Oscillator
        phases = self.network.phases(iteration)
        for oscillator, phase in zip(self.oscillators, phases):
            value = 0.5*(1+np.cos(phase))
            oscillator.set_facecolor(self.cmap_phases(value))

        # Contacts sensors
        for sensor_i, contact in enumerate(self.contact_sensors):
            value = np.clip(
                np.linalg.norm(
                    self.data.sensors.contacts.total(iteration, sensor_i)
                ),
                0, self.cmap_contact_max,
            )
            contact.set_facecolor(self.cmap_contacts(value/self.cmap_contact_max))

        # Hydros sensors
        for sensor_i, hydr in enumerate(self.hydro_sensors):
            value = np.clip(
                np.linalg.norm(
                    self.data.sensors.hydrodynamics.force(iteration, sensor_i)
                ),
                0, self.cmap_hydro_max,
            )
            hydr.set_facecolor(self.cmap_hydro(value/self.cmap_hydro_max))

        return self.animation_elements()

    def plot(self, **kwargs):
        """Plot"""
        n_oscillators = 2*self.morphology.n_joints_body

        # Options
        offset = kwargs.pop('offset', 1)
        radius = kwargs.pop('radius', 0.3)
        margin_x = kwargs.pop('margin_x', 2)
        margin_y = kwargs.pop('margin_y', 7)
        alpha = kwargs.pop('alpha', 0.3)
        title = kwargs.pop('title', 'Network')
        show_title = kwargs.pop('show_title', True)
        rads = kwargs.pop('rads', [0.2, 0.0, 0.0])
        use_colorbar = kwargs.pop('use_colorbar', False)

        # Create figure
        self.figure = plt.figure(num=title, figsize=(12, 10))
        self.axes = plt.gca()
        self.axes.cla()
        if show_title:
            plt.title(title)
        self.axes.set_xlim((-margin_x, n_oscillators-1+margin_x))
        self.axes.set_ylim((-offset-margin_y, offset+margin_y))
        self.axes.set_aspect('equal', adjustable='box')
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)
        # plt.tight_layout()

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
                self.data.network.osc_connectivity.connections.array,
                self.data.network.osc_connectivity.weights.array,
                self.data.network.osc_connectivity.desired_phases.array,
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
        self.oscillators, oscillators_texts, oscillators_connectivity = draw_network(
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
                self.data.network.contacts_connectivity.connections.array,
                self.data.network.contacts_connectivity.weights.array,
            )
            if contact_conn_cond(connection[0], connection[1])
        ]) if self.data.network.contacts_connectivity.connections.array else np.empty(0)
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
        self.contact_sensors, contact_sensor_texts, contact_connectivity = draw_network(
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
            [connection[0], connection[1], connection[2], weight]
            for connection, weight in zip(
                self.data.network.hydro_connectivity.connections.array,
                self.data.network.hydro_connectivity.weights.array,
            )
            if hydro_conn_cond(connection[0], connection[1])
        ]) if self.data.network.hydro_connectivity.connections.array else np.empty(0)
        options = {}
        hydro_frequency_weights = kwargs.pop('hydro_frequency_weights', False)
        hydro_amplitude_weights = kwargs.pop('hydro_amplitude_weights', False)
        use_weights = (
            use_colorbar
            and (hydro_frequency_weights or hydro_amplitude_weights)
        )
        if use_weights:
            if connections.any():
                options['weights'] = [
                    connection[3]
                    for connection in connections
                    if (
                        hydro_frequency_weights
                        and connection[2] == ConnectionType.LATERAL2FREQ
                    ) or (
                        hydro_amplitude_weights
                        and connection[2] == ConnectionType.LATERAL2AMP
                    )
                ]
                vmin = np.min(options['weights'])
                vmax = np.max(options['weights'])
            else:
                options['weights'] = []
                vmin, vmax = 0, 1
        self.hydro_sensors, hydro_sensor_texts, hydro_connectivity = draw_network(
            source=hydrodynamics_positions,
            destination=oscillator_positions,
            radius=radius,
            connectivity=[
                connection
                for connection in self.data.network.hydro_connectivity.connections.array
                # if hydro_conn_cond(connection[0], connection[1])
                if (not hydro_frequency_weights and not hydro_amplitude_weights)
                or (
                    hydro_frequency_weights
                    and connection[2] == ConnectionType.LATERAL2FREQ
                ) or (
                    hydro_amplitude_weights
                    and connection[2] == ConnectionType.LATERAL2AMP
                )
            ] if self.data.network.hydro_connectivity.connections.array else [],
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
            show_oscillators_connectivity,
            show_contacts_connectivity,
            show_hydrodynamics_connectivity,
        ] = [
            kwargs.pop(key, True)
            for key in [
                'show_oscillators',
                'show_contacts',
                'show_hydrodynamics',
                'show_oscillators_connectivity',
                'show_contacts_connectivity',
                'show_hydrodynamics_connectivity',
            ]
        ]
        if show_oscillators_connectivity:
            for arrow in oscillators_connectivity:
                self.axes.add_artist(arrow)
        if show_contacts_connectivity:
            for arrow in contact_connectivity:
                self.axes.add_artist(arrow)
        if show_hydrodynamics_connectivity:
            for arrow in hydro_connectivity:
                self.axes.add_artist(arrow)
        if show_oscillators:
            for circle, text in zip(self.oscillators, oscillators_texts):
                self.axes.add_artist(circle)
                self.axes.add_artist(text)
        if show_contacts:
            for circle, text in zip(self.contact_sensors, contact_sensor_texts):
                self.axes.add_artist(circle)
                self.axes.add_artist(text)
        if show_hydrodynamics:
            for circle, text in zip(self.hydro_sensors, hydro_sensor_texts):
                self.axes.add_artist(circle)
                self.axes.add_artist(text)
        if use_colorbar:
            pylog.debug('{}: {}, {}'.format(title, vmin, vmax))
            create_colorbar(self.axes, cmap, vmin, vmax)


def plot_networks_maps(morphology, data, show_all=False):
    """Plot network maps"""
    # Plot tools
    convention = AmphibiousConvention(**morphology)

    # Plot network
    network_anim = NetworkFigure(morphology, data)
    network_anim.plot(title='Complete network', show_title=False)
    network_anim.animate()

    if show_all:

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

        # Plot network oscillator weights connectivity
        network = NetworkFigure(morphology, data)
        network.plot(
            title='Oscillators complete connectivity',
            show_contacts_connectivity=False,
            show_hydrodynamics_connectivity=False,
            use_colorbar=True,
            oscillator_weights=True,
        )
        network.plot(
            title='Oscillators body2body connectivity',
            show_contacts_connectivity=False,
            show_hydrodynamics_connectivity=False,
            osc_conn_cond=body2body,
            use_colorbar=True,
            oscillator_weights=True,
        )
        network.plot(
            title='Oscillators body2limb connectivity',
            show_contacts_connectivity=False,
            show_hydrodynamics_connectivity=False,
            osc_conn_cond=body2leg,
            use_colorbar=True,
            oscillator_weights=True,
        )
        network.plot(
            title='Oscillators limb2body connectivity',
            show_contacts_connectivity=False,
            show_hydrodynamics_connectivity=False,
            osc_conn_cond=leg2body,
            rads=[0.05, 0.0, 0.0],
            use_colorbar=True,
            oscillator_weights=True,
        )
        network.plot(
            title='Oscillators limb2limb connectivity',
            show_contacts_connectivity=False,
            show_hydrodynamics_connectivity=False,
            osc_conn_cond=leg2leg,
            rads=[0.05, 0.0, 0.0],
            use_colorbar=True,
            oscillator_weights=True,
        )
        network.plot(
            title='Oscillators intralimb connectivity',
            show_contacts_connectivity=False,
            show_hydrodynamics_connectivity=False,
            osc_conn_cond=leg2sameleg,
            rads=[0.2, 0.0, 0.0],
            use_colorbar=True,
            oscillator_weights=True,
        )
        network.plot(
            title='Oscillators interlimb connectivity',
            show_contacts_connectivity=False,
            show_hydrodynamics_connectivity=False,
            osc_conn_cond=leg2diffleg,
            rads=[0.05, 0.0, 0.0],
            use_colorbar=True,
            oscillator_weights=True,
        )

        # Plot network oscillator phases connectivity
        network.plot(
            title='Oscillators complete phases',
            show_contacts_connectivity=False,
            show_hydrodynamics_connectivity=False,
            use_colorbar=True,
            oscillator_phases=True,
        )
        network.plot(
            title='Oscillators body2body phases',
            show_contacts_connectivity=False,
            show_hydrodynamics_connectivity=False,
            osc_conn_cond=body2body,
            use_colorbar=True,
            oscillator_phases=True,
        )
        network.plot(
            title='Oscillators body2limb phases',
            show_contacts_connectivity=False,
            show_hydrodynamics_connectivity=False,
            osc_conn_cond=body2leg,
            use_colorbar=True,
            oscillator_phases=True,
        )
        network.plot(
            title='Oscillators limb2body phases',
            show_contacts_connectivity=False,
            show_hydrodynamics_connectivity=False,
            osc_conn_cond=leg2body,
            rads=[0.05, 0.0, 0.0],
            use_colorbar=True,
            oscillator_phases=True,
        )
        network.plot(
            title='Oscillators limb2limb phases',
            show_contacts_connectivity=False,
            show_hydrodynamics_connectivity=False,
            osc_conn_cond=leg2leg,
            rads=[0.05, 0.0, 0.0],
            use_colorbar=True,
            oscillator_phases=True,
        )
        network.plot(
            title='Oscillators intralimb phases',
            show_contacts_connectivity=False,
            show_hydrodynamics_connectivity=False,
            osc_conn_cond=leg2sameleg,
            rads=[0.2, 0.0, 0.0],
            use_colorbar=True,
            oscillator_phases=True,
        )
        network.plot(
            title='Oscillators interlimb phases',
            show_contacts_connectivity=False,
            show_hydrodynamics_connectivity=False,
            osc_conn_cond=leg2diffleg,
            rads=[0.05, 0.0, 0.0],
            use_colorbar=True,
            oscillator_phases=True,
        )

        # Plot contacts connectivity
        network.plot(
            title='Contacts complete connectivity',
            show_oscillators_connectivity=False,
            show_hydrodynamics_connectivity=False,
            use_colorbar=True,
            contacts_weights=True,
        )
        network.plot(
            title='Contacts intralimb connectivity',
            show_oscillators_connectivity=False,
            show_hydrodynamics_connectivity=False,
            contact_conn_cond=contact2sameleg,
            use_colorbar=True,
            contacts_weights=True,
        )
        network.plot(
            title='Contacts interlimb connectivity',
            show_oscillators_connectivity=False,
            show_hydrodynamics_connectivity=False,
            contact_conn_cond=contact2diffleg,
            use_colorbar=True,
            contacts_weights=True,
        )

        # Plot hydrodynamics connectivity
        network.plot(
            title='Hydrodynamics complete connectivity',
            show_oscillators_connectivity=False,
            show_contacts_connectivity=False,
        )
        network.plot(
            title='Hydrodynamics frequency connectivity',
            show_oscillators_connectivity=False,
            show_contacts_connectivity=False,
            use_colorbar=True,
            hydro_frequency_weights=True,
        )
        network.plot(
            title='Hydrodynamics amplitude connectivity',
            show_oscillators_connectivity=False,
            show_contacts_connectivity=False,
            use_colorbar=True,
            hydro_amplitude_weights=True,
        )

    return network_anim
