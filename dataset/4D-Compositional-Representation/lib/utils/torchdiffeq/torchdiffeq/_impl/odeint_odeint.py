def odeint(func, y0, t, rtol=1e-07, atol=1e-09, method=None, options=None,
    f_options=None):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor of any shape.

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

    Args:
        func: Function that maps a Tensor holding the state `y` and a scalar Tensor
            `t` into a Tensor of state derivatives with respect to time.
        y0: N-D Tensor giving starting value of `y` at time point `t[0]`. May
            have any floating point or complex dtype.
        t: 1-D Tensor holding a sequence of time points for which to solve for
            `y`. The initial time point should be the first element of this sequence,
            and each time must be larger than the previous time. May have any floating
            point dtype. Converted to a Tensor with float64 dtype.
        rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        method: optional string indicating the integration method to use.
        options: optional dict of configuring options for the indicated integration
            method. Can only be provided if a `method` is explicitly set.
        f_options: optional dict of additional arguments for func

    Returns:
        y: Tensor, where the first dimension corresponds to different
            time points. Contains the solved value of y for each desired time point in
            `t`, with the initial value `y0` being the first element along the first
            dimension.

    Raises:
        ValueError: if an invalid `method` is provided.
        TypeError: if `options` is supplied without `method`, or if `t` or `y0` has
            an invalid dtype.
    """
    tensor_input, func, y0, t = _check_inputs(func, y0, t, f_options)
    if options is None:
        options = {}
    elif method is None:
        raise ValueError('cannot supply `options` without specifying `method`')
    if method is None:
        method = 'dopri5'
    if f_options is None:
        f_options = {}
    solver = SOLVERS[method](func, y0, f_options, rtol=rtol, atol=atol, **
        options)
    solution = solver.integrate(t)
    if tensor_input:
        solution = solution[0]
    return solution
