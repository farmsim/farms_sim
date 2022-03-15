"""Network"""

from typing import Callable
import numpy as np
from farms_data.amphibious.data import AmphibiousData
from scipy import integrate
from .network_cy import NetworkCy
from .ode import ode_oscillators_sparse


class NetworkODE(NetworkCy):
    """NetworkODE"""

    def __init__(self, data, **kwargs):
        super().__init__(
            data=data,
            dstate=np.zeros_like(data.state.array[0, :]),
        )
        self.ode: Callable = kwargs.pop('ode', ode_oscillators_sparse)
        self.data: AmphibiousData = data

        # Adaptive timestep parameters
        self.n_iterations: int = np.shape(self.state_array)[0]
        self.solver = integrate.ode(f=self.ode)
        self.solver.set_integrator('dopri5', nsteps=kwargs.pop('nsteps', 100))
        self.solver.set_initial_value(y=self.state_array[0, :], t=0.0)
        assert not kwargs, kwargs

    def step(
            self,
            iteration: int,
            time: float,
            timestep: float,
            checks: bool = False,
    ):
        """Control step"""
        if iteration == 0:
            self.drives_array[1] = self.drives_array[0]
            return
        if checks:
            assert np.array_equal(
                self.solver.y,
                self.state_array[iteration, :]
            )
        self.solver.set_f_params(self.dstate, iteration, self.data)
        self.state_array[iteration, :] = (
            self.solver.integrate(time+timestep, step=True)
        )
        if iteration < self.n_iterations-1:
            self.drives_array[iteration+1] = self.drives_array[iteration]
        if checks:
            assert self.solver.successful()
            assert abs(time+timestep-self.solver.t) < 1e-6*timestep, (
                'ODE solver time: '
                f'{self.solver.t} [s] != Simulation time: {time+timestep} [s]'
            )
