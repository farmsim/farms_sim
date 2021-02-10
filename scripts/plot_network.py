"""Plot network"""

import os
import argparse

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import farms_pylog as pylog
from farms_data.amphibious.animat_data import AnimatData
from farms_bullet.simulation.options import SimulationOptions
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.utils.network import plot_networks_maps


def parse_args():
    """Parse args"""
    parser = argparse.ArgumentParser(
        description='Plot amphibious network',
        formatter_class=(
            lambda prog:
            argparse.HelpFormatter(prog, max_help_position=50)
        ),
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data_options.yaml',
        help='Data',
    )
    parser.add_argument(
        '--animat',
        type=str,
        default='animat_options.yaml',
        help='Animat options',
    )
    parser.add_argument(
        '--simulation',
        type=str,
        default='simulation.hdf5',
        help='Simulation options',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='network.mp4',
        help='Output path',
    )
    return parser.parse_args()


class FasterFFMpegWriter(animation.FFMpegWriter):
    """FFMpeg-pipe writer bypassing figure.savefig"""

    def __init__(self, anim, **kwargs):
        super(FasterFFMpegWriter, self).__init__(**kwargs)
        self.frame_format = 'argb'
        # self.anim = anim
        # self.anim.animation_init()
        # self.frame = 0

    def grab_frame(self, **kwargs):
        """Grab frame"""
        self.fig.set_size_inches(self._w, self._h)
        self.fig.set_dpi(self.dpi)
        # if self.frame == 0:
        self.fig.canvas.draw()
        # self.anim.animation_update(self.frame)
        self._frame_sink().write(self.fig.canvas.tostring_argb())
        # self.frame += 1


def main():
    """Main"""
    args = parse_args()
    animat_options = AmphibiousOptions.load(args.animat)
    simulation_options = SimulationOptions.load(args.simulation)
    data = AnimatData.from_file(args.data)
    network_anim = plot_networks_maps(animat_options.morphology, data)
    fig = plt.gcf()
    fig.tight_layout()
    fig.set_size_inches(14, 7)
    # plt.show()

    # writer = FasterFFMpegWriter(  # ffmpegwriter
    #     anim=network_anim,
    #     fps=1/(1e-3*network_anim.interval),
    #     # codec="libx264",
    #     metadata = dict(
    #         title='FARMS network',
    #         artist='FARMS',
    #         comment='FARMS network',
    #     ),
    #     extra_args=['-vcodec', 'libx264'],
    # )
    # network_anim.animation_init()
    # with writer.saving(fig, args.output, dpi=200):
    #     for frame in range(400):
    #         print(frame)
    #         network_anim.animation_update(frame)
    #         writer.grab_frame()

    pylog.info('Saving to {}'.format(args.output))
    ffmpegwriter = animation.writers['ffmpeg']
    network_anim.animation.save(
        args.output,
        writer=FasterFFMpegWriter(  # ffmpegwriter
            # width=20, height=10,
            anim=network_anim,
            fps=1/(1e-3*network_anim.interval),
            # codec="libx264",
            metadata = dict(
                title='FARMS network',
                artist='FARMS',
                comment='FARMS network',
            ),
            extra_args=['-vcodec', 'libx264'],
            # bitrate=1800,
        ),
        dpi=200,
        # writer='pillow',
        progress_callback=lambda i, n: print(i),
    )
    pylog.info('Saved to {}'.format(args.output))


if __name__ == '__main__':
    main()
