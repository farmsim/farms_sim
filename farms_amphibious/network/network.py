"""Network"""

import numpy as np
from scipy import integrate
from ..controllers.controller import ode_oscillators_sparse


class NetworkODE:
    """NetworkODE"""

    def __init__(self, data):
        super(NetworkODE, self).__init__()
        self.ode = ode_oscillators_sparse
        self.data = data
        self.n_oscillators = data.state.n_oscillators

        # Adaptive timestep parameters
        self.solver = integrate.ode(f=self.ode)  # , jac=self.jac
        self.solver.set_integrator("dopri5")
        self.solver.set_f_params(self.data)
        self._time = 0

    def control_step(self, iteration, time, timestep):
        """Control step"""
        self.data.iteration = iteration
        self.solver.set_initial_value(
            self.data.state.array[iteration, 0, :],
            time
        )
        self.data.state.array[iteration+1, 0, :] = (
            self.solver.integrate(time+timestep)
        )

    def phases(self):
        """Oscillators phases"""
        return self.data.state.array[:, 0, :self.n_oscillators]

    def dphases(self):
        """Oscillators phases velocity"""
        return self.data.state.array[:, 1, :self.n_oscillators]

    def amplitudes(self):
        """Amplitudes"""
        return self.data.state.array[:, 0, self.n_oscillators:2*self.n_oscillators]

    def damplitudes(self):
        """Amplitudes velocity"""
        return self.data.state.array[:, 1, self.n_oscillators:2*self.n_oscillators]

    def get_outputs(self, iteration):
        """Outputs"""
        return self.amplitudes()[iteration]*(
            1 + np.cos(self.phases()[iteration])
        )

    def get_outputs_all(self):
        """Outputs"""
        return self.amplitudes()*(
            1 + np.cos(self.phases())
        )

    def get_doutputs(self, iteration):
        """Outputs velocity"""
        return self.damplitudes()[iteration]*(
            1 + np.cos(self.phases()[iteration])
        ) - (
            self.amplitudes()[iteration]
            *np.sin(self.phases()[iteration])
            *self.dphases()[iteration]
        )

    def get_doutputs_all(self):
        """Outputs velocity"""
        return self.damplitudes()*(
            1 + np.cos(self.phases)
        ) - self.amplitudes*np.sin(self.phases)*self.dphases

    def offsets(self):
        """Offset"""
        return self.data.state.array[:, 0, 2*self.n_oscillators:]

    def doffsets(self):
        """Offset velocity"""
        return self.data.state.array[:, 1, 2*self.n_oscillators:]
