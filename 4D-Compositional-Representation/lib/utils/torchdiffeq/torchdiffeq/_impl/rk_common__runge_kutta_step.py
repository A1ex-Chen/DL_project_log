def _runge_kutta_step(func, y0, f0, t0, dt, tableau, f_options):
    """Take an arbitrary Runge-Kutta step and estimate error.

    Args:
        func: Function to evaluate like `func(t, y)` to compute the time derivative
            of `y`.
        y0: Tensor initial value for the state.
        f0: Tensor initial value for the derivative, computed from `func(t0, y0)`.
        t0: float64 scalar Tensor giving the initial time.
        dt: float64 scalar Tensor giving the size of the desired time step.
        tableau: optional _ButcherTableau describing how to take the Runge-Kutta
            step.
        f_options: additional keyworded arguments for func
        name: optional name for the operation.

    Returns:
        Tuple `(y1, f1, y1_error, k)` giving the estimated function value after
        the Runge-Kutta step at `t1 = t0 + dt`, the derivative of the state at `t1`,
        estimated error at `t1`, and a list of Runge-Kutta coefficients `k` used for
        calculating these terms.
    """
    dtype = y0[0].dtype
    device = y0[0].device
    t0 = _convert_to_tensor(t0, dtype=dtype, device=device)
    dt = _convert_to_tensor(dt, dtype=dtype, device=device)
    k = tuple(map(lambda x: [x], f0))
    for alpha_i, beta_i in zip(tableau.alpha, tableau.beta):
        ti = t0 + alpha_i * dt
        yi = tuple(y0_ + _scaled_dot_product(dt, beta_i, k_) for y0_, k_ in
            zip(y0, k))
        tuple(k_.append(f_) for k_, f_ in zip(k, func(ti, yi, **f_options)))
    if not (tableau.c_sol[-1] == 0 and tableau.c_sol[:-1] == tableau.beta[-1]):
        yi = tuple(y0_ + _scaled_dot_product(dt, tableau.c_sol, k_) for y0_,
            k_ in zip(y0, k))
    y1 = yi
    f1 = tuple(k_[-1] for k_ in k)
    y1_error = tuple(_scaled_dot_product(dt, tableau.c_error, k_) for k_ in k)
    return y1, f1, y1_error, k