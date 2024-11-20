def rk4_alt_step_func(func, t, dt, y, f_options, k1=None):
    """Smaller error with slightly more compute."""
    if k1 is None:
        k1 = func(t, y, **f_options)
    k2 = func(t + dt / 3, tuple(y_ + dt * k1_ / 3 for y_, k1_ in zip(y, k1)
        ), **f_options)
    k3 = func(t + dt * 2 / 3, tuple(y_ + dt * (k1_ / -3 + k2_) for y_, k1_,
        k2_ in zip(y, k1, k2)), **f_options)
    k4 = func(t + dt, tuple(y_ + dt * (k1_ - k2_ + k3_) for y_, k1_, k2_,
        k3_ in zip(y, k1, k2, k3)), **f_options)
    return tuple((k1_ + 3 * k2_ + 3 * k3_ + k4_) * (dt / 8) for k1_, k2_,
        k3_, k4_ in zip(k1, k2, k3, k4))
