"""Network"""

import numpy as np
from scipy import integrate
from ..model.data import DTYPE
from ..controllers.controller import (
    ode_oscillators_sparse,
    ode_dphase,
    ode_damplitude,
    ode_contacts,
    ode_hydro,
    ode_joints,
    rk_oscillators,
)


def ode_oscillators_sparse_python(time, state, data, network):
    """ODE"""
    state = np.array(state, dtype=DTYPE)
    ode_dphase(
        state=state,
        dstate=data.state.array[data.iteration][1],
        oscillators=data.network.oscillators,
        connectivity=data.network.connectivity,
    )
    ode_damplitude(
        state=state,
        dstate=data.state.array[data.iteration][1],
        oscillators=data.network.oscillators,
    )
    ode_contacts(
        iteration=data.iteration,
        state=state,
        dstate=data.state.array[data.iteration][1],
        contacts=data.sensors.contacts,
        contacts_connectivity=data.network.contacts_connectivity,
    )
    ode_hydro(
        iteration=data.iteration,
        state=state,
        dstate=data.state.array[data.iteration][1],
        hydrodynamics=data.sensors.hydrodynamics,
        hydro_connectivity=data.network.hydro_connectivity,
        n_oscillators=data.network.oscillators.size[1],
    )
    ode_joints(
        state=state,
        dstate=data.state.array[data.iteration][1],
        joints=data.joints,
        n_oscillators=data.network.oscillators.size[1],
    )
    return data.state.array[data.iteration][1]


def ode_oscillators_sparse_python2(time, state, data, network):
    """ODE"""
    ode_oscillators_sparse_python(time, state.astype(DTYPE), data, network)


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
        self.solver.set_f_params(self.data, self.data.network)
        self.solver.set_initial_value(y=self.data.state.array[0, 0, :], t=0.0)

        # # Solver
        # self.solver_fix = integrate.RK45(
        #     fun=self.fun,
        #     t0=0,
        #     y0=self.data.state.array[0, 0],
        #     t_bound=1000,
        #     max_step=1e-4,
        #     first_step=1e-4,
        #     rtol=1e-6,
        #     atol=1e-8,
        # )

    # def fun(self, time, state):
    #     """ODE"""
    #     return self.ode(time, state, self.data, self.data.network)

    def control_step(self, iteration, time, timestep):
        """Control step"""
        self.data.iteration = iteration
        self.data.state.array[iteration+1, 0, :] = (
            self.solver.integrate(time+timestep)
        )
        assert self.solver.successful()
        assert abs(time+timestep-self.solver.t) < 1e-6*timestep, (
            'ODE solver time: {} [s] != Simulation time: {} [s]'.format(
                self.solver.t,
                time+timestep,
            )
        )

        # # Runge-Kutta
        # state = self.data.state.array[iteration, 0, :]
        # size = len(state)
        # n_iterations = 50
        # k1 = np.zeros(size)
        # k2 = np.zeros(size)
        # k3 = np.zeros(size)
        # k4 = np.zeros(size)
        # _state = np.zeros(size)
        # _state = np.zeros(size)
        # self.data.state.array[iteration+1, 0, :] = (
        #     self.data.state.array[iteration, 0, :]
        # )
        # for i in range(n_iterations):
        #     rk_oscillators(
        #         time=time+i*n_iterations,
        #         timestep=timestep/n_iterations,
        #         state=np.copy(self.data.state.array[iteration+1, 0, :]),
        #         data=self.data,
        #         network=self.data.network,
        #         k1=k1,
        #         k2=k2,
        #         k3=k3,
        #         k4=k4,
        #         state_out=self.data.state.array[iteration+1, 0, :],
        #     )

        # # Solve_ivp
        # self.data.state.array[iteration+1, 0, :] = integrate.solve_ivp(
        #     self.ode,
        #     [0, timestep],
        #     self.data.state.array[iteration, 0, :],
        #     method='RK23',
        #     t_eval=[timestep],
        #     args=(self.data, self.data.network),
        # )['y'][0]

        # # Scipy RK45
        # self.solver_fix.t = time
        # while self.solver_fix.t < time+timestep:
        #     self.solver_fix.step()
        #     print(self.solver_fix.t)
        # self.data.state.array[iteration+1, 0, :] = self.solver_fix.y

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
