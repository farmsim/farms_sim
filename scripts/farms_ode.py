"""Test ODE solving"""

import time
import numpy as np
import sympy as sp
from sympy.utilities.autowrap import autowrap
from sympy.utilities.codegen import codegen


def solve_ode():
    """Solve ODE"""
    t = sp.Symbol("t")
    f = sp.Symbol("f")
    x0 = sp.Symbol("x0")
    phi = sp.Symbol("phi")
    x = sp.Function("x")
    xt = x(t)
    dxdt = xt.diff(t)
    # ode = sp.Eq(dxdt, f+sp.sin(-xt+phi))
    ode = sp.Eq(dxdt, f-xt+phi)
    print("ODE:")
    sp.pretty_print(ode)
    res = sp.dsolve(
        ode,
        xt,
        ics = {x(0): x0},
        simplify=True
    )
    print("Solved ODE:")
    sp.pretty_print(res)

    print("Lambdifying")
    sol = sp.lambdify((x0, t, f, phi), res.args[1])
    step = sol(x0, 1e-3, f, phi)
    sp.pretty_print(step)
    sol = sp.lambdify((xt, t, f, phi), res)


def solve_ode2():
    """Solve ODE 2"""
    size = 3
    t = sp.Symbol("t")
    dt = sp.Symbol("dt")
    x = sp.MatrixSymbol("x", size, 1)
    f = sp.MatrixSymbol("f", size, 1)
    weights = sp.MatrixSymbol("W", size, size)
    phi = sp.MatrixSymbol("phi", size, size)
    # ODE
    print("ODE:")
    ode = f - sp.Matrix([
        sum([
            (
                weights[i, j]
            )*sp.sin(x[i] - x[j] - phi[i, j])
            for j in range(size)
        ])
        for i in range(size)
    ])
    sp.pretty_print(sp.Matrix(ode))
    # DODE
    print("DODE")
    dode = sp.Matrix([ode[i, 0].diff(t) for i in range(size)])
    sp.pretty_print(dode)
    fode = sp.lambdify((x, f, weights, phi), ode, ["sympy"])
    fdode = sp.lambdify((x, f, weights, phi), dode, ["sympy"])
    print("Euler:")
    xi = x
    for _ in range(3):
        xi += (
            dt*fode(xi, f, weights, phi)
            + 0.5*dt**2*fdode(xi, f, weights, phi)
        )
    sp.pretty_print(sp.Matrix(xi))
    # xi_simple = sp.simplify(xi)
    # sp.pretty_print(xi_simple)
    xi_cse = sp.cse(sp.Matrix(xi))
    print("CSE variables:")
    for cse in xi_cse[0]:
        sp.pretty_print(cse)
    print("CSE equation:")
    for cse in xi_cse[1]:
        sp.pretty_print(cse)


def odefun(t_n, phases, freqs, weights, phi, size):
    """ODE function"""
    return freqs - sp.Matrix([
        sum([
            (
                weights[i, j]
            )*sp.sin(phases[i] - phases[j] - phi[i, j])
            for j in range(size)
        ])
        for i in range(size)
    ])


def rk4(fun, timestep, t_n, state, *fun_params):
    """Runge-Kutta step integration"""
    k_1 = timestep*fun(t_n, state, *fun_params)
    print("Computed k1")
    k_2 = timestep*fun(t_n+timestep/2, state+k_1/2, *fun_params)
    print("Computed k2")
    k_3 = timestep*fun(t_n+timestep/2, state+k_2/2, *fun_params)
    print("Computed k3")
    k_4 = timestep*fun(t_n+timestep, state+k_3, *fun_params)
    print("Computed k4")
    return (k_1+2*k_2+2*k_3+k_4)/6


def solve_ode_rk():
    """Solve ODE 2"""
    size = 5
    t = sp.Symbol("t")
    dt = sp.Symbol("dt")
    x_sym = sp.MatrixSymbol("x", size, 1)
    f_sym = sp.MatrixSymbol("f", size, 1)
    weights_sym = sp.MatrixSymbol("W", size, size)
    phi_sym = sp.MatrixSymbol("phi", size, size)
    # ODE
    # print("ODE:")
    # ode = odefun(x_sym, f_sym, weights_sym, phi_sym, size)
    # sp.pretty_print(sp.Matrix(ode))
    # fode = sp.lambdify(
    #     (x_sym, f_sym, weights_sym, phi_sym),
    #     sp.Matrix(ode),
    #     ["sympy"]
    # )
    # dt = 1e-3
    x = sp.Matrix(x_sym)
    f = sp.Matrix(f_sym)
    weights = sp.Matrix(weights_sym)
    phi = sp.Matrix(phi_sym)
    # x, f, weights, phi = x_sym, f_sym, weights_sym, phi_sym
    print("Computing Runge-Kutta")
    _ode = sp.lambdify(
        args=(sp.Symbol("time"), x_sym, f_sym, weights_sym, phi_sym),
        expr=odefun(0, x, f, weights, phi, size),
        modules="sympy"
    )
    sp.pretty_print(_ode)
    xp = rk4(_ode, dt, 0, x, f, weights, phi)
    print("Computed x_plus")
    sp.pretty_print(xp)
    # print("Computing CSE")
    # xi_cse = sp.cse(sp.Matrix(xp))
    # print("CSE variables:")
    # for cse in xi_cse[0]:
    #     sp.pretty_print(cse)
    # print("CSE equation:")
    # for cse in xi_cse[1]:
    #     sp.pretty_print(cse)

    # Generate code (Can also use F95)
    # [(c_name, c_code), (h_name, c_header)] = (
    #     codegen(("wrapper_code", xp), "C99", "test", header=True, empty=False)
    # )
    # print(c_code)

    print("Wrapping")
    ode_fast = autowrap(
        xp,
        args=(x_sym, f_sym, weights_sym, phi_sym, dt),
        tempdir="./temp_ode",
        language="C",
        backend="cython",
        verbose=True
    )
    tic = time.time()
    res = ode_fast(
        np.ones([size, 1]),
        np.ones([size, 1]),
        np.ones([size, size]),
        np.ones([size, size]),
        1e-3
    )
    print("Time: {} [s]".format(time.time() - tic))
    print(res)


def tensor():
    """Solve ODE with tensor"""
    from sympy.utilities.autowrap import autowrap
    from sympy import symbols, IndexedBase, Idx, Eq
    size = 3
    m, n = symbols('m n', integer=True)
    # i = Idx('i', m)
    # j = Idx('j', n)
    i = Idx('i', size)
    j = Idx('j', size)
    # i = Idx('i', (0, 10))
    # j = Idx('j', (0, 10))
    dx = sp.IndexedBase("dx")
    x = sp.IndexedBase("x")
    f = sp.IndexedBase("f")
    W = sp.IndexedBase("W")
    phi = sp.IndexedBase("phi")
    equation = f[i] + sp.Sum(
        W[i, j]*sp.sin(x[i] - x[j] - phi[i, j]),
        (j, 0, size-1)
    )
    sp.pretty_print(sp.Array(equation))
    equation_ev = sp.Matrix(
        [
            sum([
                equation.doit().subs({i: ni, j: nj, n: size, m: size})
                for nj in range(size)
            ]) for ni in range(size)
        ]
    )
    sp.pretty_print(equation_ev)
    time_n = sp.Symbol("t")
    timestep = sp.Symbol("dt")
    ode_ev = sp.lambdify(
        (time_n, x, f, W, phi),
        equation_ev,
        "sympy"
    )
    x_sym = sp.MatrixSymbol("x", size, 1)
    f_sym = sp.MatrixSymbol("f", size, 1)
    weights_sym = sp.MatrixSymbol("W", size, size)
    phi_sym = sp.MatrixSymbol("phi", size, size)
    fun = ode_ev(
        timestep,
        sp.Matrix(x_sym),
        sp.Matrix(f_sym),
        sp.Matrix(weights_sym),
        sp.Matrix(phi_sym)
    )
    sp.pretty_print(fun)
    _rk = rk4(
        ode_ev,
        timestep,
        time_n,
        sp.Matrix(x_sym),
        sp.Matrix(f_sym),
        sp.Matrix(weights_sym),
        sp.Matrix(phi_sym)
    )
    print("Wrapping")
    ode_fast = autowrap(
        _rk,
        args=(timestep, x_sym, f_sym, weights_sym, phi_sym),
        tempdir="./temp_ode",
        # language="C",
        # backend="cython",
        verbose=True
    )
    # print(ode_ev(
    #     np.ones([size, 1]),
    #     np.ones([size, 1]),
    #     np.ones([size, size]),
    #     np.ones([size, size])
    # ))
    # x_sym = sp.Matrix(sp.MatrixSymbol("x", size, 1))
    # dx_sym = sp.Matrix(sp.MatrixSymbol("dx", size, 1))
    # f_sym = sp.Matrix(sp.MatrixSymbol("f", size, 1))
    # W_sym = sp.Matrix(sp.MatrixSymbol("W", size, size))
    # phi_sym = sp.Matrix(sp.MatrixSymbol("phi", size, size))
    # sp.pretty_print(equation.doit())
    # ode_ev = autowrap(
    #     equation,
    #     args=(x, f, W, phi, dx, m, n),
    #     # args=(x_sym, f_sym, W_sym, phi_sym, dx_sym, m, n),
    # )
    # ode_ev(
    #     np.ones([size, 1]),
    #     np.ones([size, 1]),
    #     np.ones([size, size]),
    #     np.ones([size, size])
    # )
    # ode_fast = autowrap(
    #     xp,
    #     args=(x_sym, f_sym, weights_sym, phi_sym, dt),
    #     tempdir="./temp_ode",
    #     # language="C",
    #     # backend="cython",
    #     verbose=True
    # )


def main():
    """Main"""
    # solve_ode()
    # solve_ode2()
    solve_ode_rk()
    # tensor()


if __name__ == '__main__':
    main()
