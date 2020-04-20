"""Interface"""

from farms_bullet.interface.interface import UserParameters, DebugParameter


class AmphibiousUserParameters(UserParameters):
    """Amphibious user parameters"""

    def __init__(self, options):
        super(AmphibiousUserParameters, self).__init__()
        self['drive_speed'] = DebugParameter(
            'Drive speed',
            options.control.network.drives_init[0],
            0.9, 5.1
        )
        self['drive_turn'] = DebugParameter(
            'Drive turn',
            options.control.network.drives_init[1],
            -0.2, 0.2
        )

    def drive_speed(self):
        """Drive speed"""
        return self['drive_speed']

    def drive_turn(self):
        """Drive turn"""
        return self['drive_turn']
