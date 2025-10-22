#!/usr/bin/env python3
"""Run a simulation with FARMS"""

import time

from farms_core import pylog
from farms_core.utils.profile import profile
from farms_core.extensions.extensions import import_item
from .utils.parse_args import sim_parse_args
from .simulation import (
    setup_from_clargs,
    run_simulation,
)


def main():
    """Main"""

    # Setup
    pylog.info('Loading options from clargs')
    _, exp_options, simulator  = setup_from_clargs()

    # Data
    experiment_data_loader = import_item(exp_options.loaders.experiment_data)
    experiment_data = experiment_data_loader.from_options(exp_options)

    # Simulation
    pylog.info('Creating simulation environment')
    run_simulation(
        experiment_data=experiment_data,
        experiment_options=exp_options,
        simulator=simulator,
    )


def profile_simulation():
    """Profile simulation"""
    tic = time.time()
    clargs = sim_parse_args()
    profile(function=main, profile_filename=clargs.profile)
    pylog.info('Total simulation time: %s [s]', time.time() - tic)


if __name__ == '__main__':
    profile_simulation()
