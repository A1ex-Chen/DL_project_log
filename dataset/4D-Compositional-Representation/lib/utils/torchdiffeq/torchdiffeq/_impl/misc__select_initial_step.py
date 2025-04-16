def _select_initial_step(fun, t0, y0, f_options, order, rtol, atol, f0=None):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t0 : float
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    f_options: dictionary
        Additional keyworded arguments for fun
    direction : float
        Integration direction.
    order : float
        Method order.
    rtol : float
        Desired relative tolerance.
    atol : float
        Desired absolute tolerance.

    Returns
    -------
    h_abs : float
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    t0 = t0.to(y0[0])
    if f0 is None:
        f0 = fun(t0, y0, **f_options)
    rtol = rtol if _is_iterable(rtol) else [rtol] * len(y0)
    atol = atol if _is_iterable(atol) else [atol] * len(y0)
    scale = tuple(atol_ + torch.abs(y0_) * rtol_ for y0_, atol_, rtol_ in
        zip(y0, atol, rtol))
    d0 = tuple(_norm(y0_ / scale_) for y0_, scale_ in zip(y0, scale))
    d1 = tuple(_norm(f0_ / scale_) for f0_, scale_ in zip(f0, scale))
    if max(d0).item() < 1e-05 or max(d1).item() < 1e-05:
        h0 = torch.tensor(1e-06).to(t0)
    else:
        h0 = 0.01 * max(d0_ / d1_ for d0_, d1_ in zip(d0, d1))
    y1 = tuple(y0_ + h0 * f0_ for y0_, f0_ in zip(y0, f0))
    f1 = fun(t0 + h0, y1, **f_options)
    d2 = tuple(_norm((f1_ - f0_) / scale_) / h0 for f1_, f0_, scale_ in zip
        (f1, f0, scale))
    if max(d1).item() <= 1e-15 and max(d2).item() <= 1e-15:
        h1 = torch.max(torch.tensor(1e-06).to(h0), h0 * 0.001)
    else:
        h1 = (0.01 / max(d1 + d2)) ** (1.0 / float(order + 1))
    return torch.min(100 * h0, h1)
