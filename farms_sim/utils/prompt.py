"""Prompt"""

import os

import matplotlib.pyplot as plt

from farms_core import pylog
from farms_core.simulation.options import Simulator
from farms_core.model.data import AnimatData


def strtobool (val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    if val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    raise ValueError(f'invalid truth value {val}')


def prompt(query, default):
    """Prompt"""
    val = input(f'{query} [{"Y/n" if default else "y/N"}]: ')
    try:
        ret = strtobool(val) if val != '' else default
    except ValueError:
        pylog.error('Did not recognise \'%s\', please reply with a y/n', val)
        return prompt(query, default)
    return ret


def prompt_postprocessing(sim, query=True, **kwargs):
    """Prompt postprocessing"""
    # Arguments
    log_path = kwargs.pop('log_path', '')
    verify = kwargs.pop('verify', False)
    extension = kwargs.pop('extension', 'pdf')
    simulator = kwargs.pop('simulator', Simulator.MUJOCO)
    data_loader = kwargs.pop('animat_data_loader', AnimatData)
    assert not kwargs, kwargs

    # Post-processing
    pylog.info('Simulation post-processing')
    save_data = (
        (query and prompt('Save data', False))
        or log_path and not query
    )
    if log_path:
        os.makedirs(log_path, exist_ok=True)
    show_plots = prompt('Show plots', False) if query else False
    iteration = (
        sim.iteration
        if simulator == Simulator.PYBULLET
        else sim.task.iteration  # Simulator.MUJOCO
    )
    sim.postprocess(
        iteration=iteration,
        log_path=log_path if save_data else '',
        plot=show_plots,
    )
    if save_data and verify:
        pylog.debug('Data saved, now loading back to check validity')
        data_loader.from_file(os.path.join(log_path, 'simulation.hdf5'))
        pylog.debug('Data successfully saved and logged back')

    # Save MuJoCo MJCF
    if save_data and simulator == Simulator.MUJOCO:
        sim.save_mjcf_xml(os.path.join(log_path, 'sim_mjcf.xml'))

    # Save plots
    if show_plots and query and prompt('Save plots', False):
        for fig in [plt.figure(num) for num in plt.get_fignums()]:
            path = os.path.join(log_path, fig.canvas.get_window_title())
            filename = f'{path}.{extension}'
            filename = filename.replace(' ', '_')
            pylog.debug('Saving to %s', filename)
            fig.savefig(filename, format=extension)

    # Show plots
    if show_plots or query and prompt('Show connectivity plots', False):
        plt.show()
