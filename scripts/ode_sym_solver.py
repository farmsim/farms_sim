"""ODE sym solver"""

import sympy as sp


def main():
    """Main

    dxdt = f(x)
    with x(t=t0) = x0

    => x(t) = ...

    x(t+dt) = f(t, x(t))

    """
    x0, t, dt = sp.symbols("x0 t dt")
    x = sp.Function("x")
    print(x)
    dxdt = sp.diff(x(t), t)
    print(dxdt)
    # fdxdt = x(t)
    fdxdt = sp.sin(x(t))
    sol = sp.dsolve(dxdt - fdxdt, ics={x(0): x0})
    print(sol)
    # fsol = sp.lambdify([t, x0], sol[1])
    # print(fsol(dt, x0))


if __name__ == '__main__':
    main()
