def _interp_evaluate(coefficients, t0, t1, t):
    """Evaluate polynomial interpolation at the given time point.

    Args:
        coefficients: list of Tensor coefficients as created by `interp_fit`.
        t0: scalar float64 Tensor giving the start of the interval.
        t1: scalar float64 Tensor giving the end of the interval.
        t: scalar float64 Tensor giving the desired interpolation point.

    Returns:
        Polynomial interpolation of the coefficients at time `t`.
    """
    dtype = coefficients[0][0].dtype
    device = coefficients[0][0].device
    t0 = _convert_to_tensor(t0, dtype=dtype, device=device)
    t1 = _convert_to_tensor(t1, dtype=dtype, device=device)
    t = _convert_to_tensor(t, dtype=dtype, device=device)
    assert (t0 <= t) & (t <= t1
        ), 'invalid interpolation, fails `t0 <= t <= t1`: {}, {}, {}'.format(t0
        , t, t1)
    x = ((t - t0) / (t1 - t0)).type(dtype).to(device)
    xs = [torch.tensor(1).type(dtype).to(device), x]
    for _ in range(2, len(coefficients)):
        xs.append(xs[-1] * x)
    return tuple(_dot_product(coefficients_, reversed(xs)) for
        coefficients_ in zip(*coefficients))
