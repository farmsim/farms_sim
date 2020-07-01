"""Interface"""

from farms_bullet.interface.interface import UserParameters, DebugParameter


class AmphibiousUserParameters(UserParameters):
    """Amphibious user parameters"""

    def __init__(self, animat_options, simulation_options):
        super(AmphibiousUserParameters, self).__init__(simulation_options)
        self['drive_speed'] = DebugParameter(
            'Drive speed',
            animat_options.control.network.drives[0].initial_value,
            0.9, 5.1
        )
        self['drive_turn'] = DebugParameter(
            'Drive turn',
            animat_options.control.network.drives[1].initial_value,
            -0.2, 0.2
        )

    def init(self):
        """Initialise"""
        super().init()
        self['drive_speed'].init()
        self['drive_turn'].init()

    def drive_speed(self):
        """Drive speed"""
        return self['drive_speed']

    def drive_turn(self):
        """Drive turn"""
        return self['drive_turn']
