"""Network"""

from typing import Callable
import numpy as np
from farms_data.amphibious.data import AmphibiousData
from scipy import integrate
from scipy.integrate._ode import ode as ODE
from .ode import ode_oscillators_sparse


class NetworkODE:
    """NetworkODE"""

    def __init__(self, data, **kwargs):
        super().__init__()
        self.dstate = np.zeros_like(data.state.array[0, :])
        self.ode: Callable = kwargs.pop('ode', ode_oscillators_sparse)
        self.data: AmphibiousData = data
        state_array = self.data.state.array
        self.n_iterations: int = np.shape(state_array)[0]
        self.solver: ODE = integrate.ode(f=self.ode)
        self.solver.set_integrator('dopri5', nsteps=kwargs.pop('nsteps', 100))
        self.solver.set_initial_value(y=state_array[0, :], t=0.0)
        assert not kwargs, kwargs

    def copy_next_drive(self, iteration):
        """Set initial drive"""
        array = self.data.network.drives.array
        array[iteration+1] = array[iteration]

    def step(
            self,
            iteration: int,
            time: float,
            timestep: float,
            checks: bool = False,
    ):
        """Control step"""
        if iteration == 0:
            self.copy_next_drive(iteration)
            return
        if checks:
            assert np.array_equal(
                self.solver.y,
                self.data.state.array[iteration, :]
            )
        self.solver.set_f_params(self.dstate, iteration, self.data)
        self.data.state.array[iteration, :] = (
            self.solver.integrate(time+timestep, step=True)
        )
        if iteration < self.n_iterations-1:
            self.copy_next_drive(iteration)
        if checks:
            assert self.solver.successful()
            assert abs(time+timestep-self.solver.t) < 1e-6*timestep, (
                'ODE solver time: '
                f'{self.solver.t} [s] != Simulation time: {time+timestep} [s]'
            )
