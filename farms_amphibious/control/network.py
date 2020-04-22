"""Network"""

import numpy as np
from scipy import integrate
from .ode import (
    ode_oscillators_sparse,
    ode_oscillators_sparse_no_sensors,
    ode_oscillators_sparse_tegotae,
)


class NetworkODE:
    """NetworkODE"""

    def __init__(self, data):
        super(NetworkODE, self).__init__()
        self.ode = ode_oscillators_sparse
        self.data = data
        self.n_oscillators = data.state.n_oscillators

        # Adaptive timestep parameters
        initial_state = self.data.state.array[0, :]
        self.solver = integrate.ode(f=self.ode)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(y=initial_state, t=0.0)
        self.dstate = np.zeros_like(initial_state)

    def step(self, iteration, time, timestep, checks=False):
        """Control step"""
        if checks:
            assert np.array_equal(
                self.solver.y,
                self.data.state.array[iteration, :]
            )
        self.solver.set_f_params(self.dstate, iteration, self.data)
        self.data.state.array[iteration+1, :] = (
            self.solver.integrate(time+timestep)
        )
        if checks:
            assert self.solver.successful()
            assert abs(time+timestep-self.solver.t) < 1e-6*timestep, (
                'ODE solver time: {} [s] != Simulation time: {} [s]'.format(
                    self.solver.t,
                    time+timestep,
                )
            )

    def phases(self, iteration=None):
        """Oscillators phases"""
        return (
            self.data.state.array[iteration, :self.n_oscillators]
            if iteration is not None else
            self.data.state.array[:, :self.n_oscillators]
        )

    def amplitudes(self, iteration=None):
        """Amplitudes"""
        return (
            self.data.state.array[
                iteration,
                self.n_oscillators:2*self.n_oscillators
            ]
            if iteration is not None else
            self.data.state.array[:, self.n_oscillators:2*self.n_oscillators]
        )

    def offsets(self, iteration=None):
        """Offset"""
        return (
            self.data.state.array[iteration, 2*self.n_oscillators:]
            if iteration is not None
            else self.data.state.array[:, 2*self.n_oscillators:]
        )

    def outputs(self, iteration=None):
        """Outputs"""
        return self.amplitudes(iteration)*(1 + np.cos(self.phases(iteration)))
