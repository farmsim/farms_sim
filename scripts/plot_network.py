"""Plot network"""

import os
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from PIL import Image
from moviepy.editor import VideoClip

import farms_pylog as pylog
from farms_data.utils.profile import profile
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


def main(use_moviepy=True):
    """Main"""
    matplotlib.use('Agg')
    # Clargs
    args = parse_args()
    # Setup
    animat_options = AmphibiousOptions.load(args.animat)
    simulation_options = SimulationOptions.load(args.simulation)
    data = AnimatData.from_file(args.data)
    network_anim = plot_networks_maps(animat_options.morphology, data)[0]
    fig = plt.gcf()
    ax = plt.gca()
    fig.tight_layout()
    fig.set_size_inches(14, 7)
    # Draw to save background
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    # Animation elements
    elements = network_anim.animation_elements()
    # Extension
    _, extension = os.path.splitext(args.output)
    if extension == '.png':
        # Save every frame
        for frame in range(network_anim.n_frames):
            pylog.debug('Saving frame {}'.format(frame))
            # Restore
            fig.canvas.restore_region(background)
            # Update
            network_anim.animation_update(frame)
            # Draw
            for element in elements:
                ax.draw_artist(element)
            # Save
            img = Image.frombytes(
                'RGB',
                fig.canvas.get_width_height(),
                fig.canvas.tostring_rgb(),
            )
            img.save(args.output.format(frame=frame))
    elif extension == '.mp4':
        if use_moviepy:
            # Use Moviepy
            fps = 1/(1e-3*network_anim.interval)
            duration = network_anim.timestep*network_anim.n_iterations
            n_frames = network_anim.n_frames
            def make_frame(t):
                """Make frame"""
                frame = min([int(t*fps), network_anim.n_frames])
                pylog.debug('Saving frame {frame}/{total}'.format(
                    frame=frame,
                    total=n_frames,
                ))
                # Restore
                fig.canvas.restore_region(background)
                # Update
                network_anim.animation_update(frame)
                # Draw
                for element in elements:
                    ax.draw_artist(element)
                return np.array(fig.canvas.renderer.buffer_rgba())[:,:,:3]
            anim = VideoClip(make_frame, duration=duration)
            anim.write_videofile(
                filename=args.output,
                fps=fps,
                codec='libx264',
                logger=None,
            )
            anim.close()
        else:
            # Use Matplotlib
            moviewriter = animation.writers['ffmpeg'](
                fps=1/(1e-3*network_anim.interval),
                metadata = dict(
                    title='FARMS network',
                    artist='FARMS',
                    comment='FARMS network',
                ),
                extra_args=['-vcodec', 'libx264'],
            )
            with moviewriter.saving(
                    fig=fig,
                    outfile=args.output,
                    dpi=100,
            ):
                for frame in range(network_anim.n_frames):
                    pylog.debug('Saving frame {}'.format(frame))
                    network_anim.animation_update(frame)
                    moviewriter.grab_frame()
    else:
        raise Exception('Unknown file extension {}'.format(extension))
    pylog.info('Saved to {}'.format(args.output))


if __name__ == '__main__':
    profile(main)
    # main()
