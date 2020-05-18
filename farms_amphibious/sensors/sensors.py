"""GPS for amphibious animat"""

from farms_bullet.sensors.sensors import LinksStatesSensor


class AmphibiousGPS(LinksStatesSensor):
    """Amphibious GPS"""

    def __init__(self, array, animat_id, links, options, units):
        super(AmphibiousGPS, self).__init__(
            array=array,
            animat_id=animat_id,
            links=links,
            units=units
        )
        self.options = options

    def update(self, iteration):
        """Update sensor"""
        self.collect(iteration, self.links)
