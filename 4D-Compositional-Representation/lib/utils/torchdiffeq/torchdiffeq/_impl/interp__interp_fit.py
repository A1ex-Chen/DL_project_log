def _interp_fit(y0, y1, y_mid, f0, f1, dt):
    """Fit coefficients for 4th order polynomial interpolation.

    Args:
        y0: function value at the start of the interval.
        y1: function value at the end of the interval.
        y_mid: function value at the mid-point of the interval.
        f0: derivative value at the start of the interval.
        f1: derivative value at the end of the interval.
        dt: width of the interval.

    Returns:
        List of coefficients `[a, b, c, d, e]` for interpolating with the polynomial
        `p = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e` for values of `x`
        between 0 (start of interval) and 1 (end of interval).
    """
    a = tuple(_dot_product([-2 * dt, 2 * dt, -8, -8, 16], [f0_, f1_, y0_,
        y1_, y_mid_]) for f0_, f1_, y0_, y1_, y_mid_ in zip(f0, f1, y0, y1,
        y_mid))
    b = tuple(_dot_product([5 * dt, -3 * dt, 18, 14, -32], [f0_, f1_, y0_,
        y1_, y_mid_]) for f0_, f1_, y0_, y1_, y_mid_ in zip(f0, f1, y0, y1,
        y_mid))
    c = tuple(_dot_product([-4 * dt, dt, -11, -5, 16], [f0_, f1_, y0_, y1_,
        y_mid_]) for f0_, f1_, y0_, y1_, y_mid_ in zip(f0, f1, y0, y1, y_mid))
    d = tuple(dt * f0_ for f0_ in f0)
    e = y0
    return [a, b, c, d, e]
