def scipy_bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
    """
    cv = np.asarray(cv)
    count = cv.shape[0]
    if periodic:
        kv = np.arange(-degree, count + degree + 1)
        factor, fraction = divmod(count + degree + 1, count)
        cv = np.roll(np.concatenate((cv,) * factor + (cv[:fraction],)), -1,
            axis=0)
        degree = np.clip(degree, 1, degree)
    else:
        degree = np.clip(degree, 1, count - 1)
        kv = np.clip(np.arange(count + degree + 1) - degree, 0, count - degree)
    max_param = count - degree * (1 - periodic)
    spl = si.BSpline(kv, cv, degree)
    return spl(np.linspace(0, max_param, n))
