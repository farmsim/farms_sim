"""Network"""

import numpy as np
from scipy import integrate
from ..controllers.controller import (
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
        self.solver = integrate.ode(f=self.ode)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(y=self.data.state.array[0, 0, :], t=0.0)

    def control_step(self, iteration, time, timestep, check=False):
        """Control step"""
        if check:
            assert np.array_equal(
                self.solver.y,
                self.data.state.array[iteration, 0, :]
            )
        self.solver.set_f_params(iteration, self.data)
        self.data.state.array[iteration+1, 0, :] = (
            self.solver.integrate(time+timestep)
        )
        if check:
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
            self.data.state.array[iteration, 0, :self.n_oscillators]
            if iteration is not None else
            self.data.state.array[:, 0, :self.n_oscillators]
        )

    def amplitudes(self, iteration=None):
        """Amplitudes"""
        return (
            self.data.state.array[
                iteration, 0,
                self.n_oscillators:2*self.n_oscillators
            ]
            if iteration is not None else
            self.data.state.array[
                :, 0,
                self.n_oscillators:2*self.n_oscillators
            ]
        )

    def offsets(self):
        """Offset"""
        return self.data.state.array[:, 0, 2*self.n_oscillators:]

    def outputs(self, iteration=None):
        """Outputs"""
        return self.amplitudes(iteration)*(1 + np.cos(self.phases(iteration)))
