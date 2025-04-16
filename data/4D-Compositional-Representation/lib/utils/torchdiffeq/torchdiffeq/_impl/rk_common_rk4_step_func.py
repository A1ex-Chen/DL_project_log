def rk4_step_func(func, t, dt, y, f_options, k1=None):
    if k1 is None:
        k1 = func(t, y, **f_options)
    k2 = func(t + dt / 2, tuple(y_ + dt * k1_ / 2 for y_, k1_ in zip(y, k1)
        ), **f_options)
    k3 = func(t + dt / 2, tuple(y_ + dt * k2_ / 2 for y_, k2_ in zip(y, k2)
        ), **f_options)
    k4 = func(t + dt, tuple(y_ + dt * k3_ for y_, k3_ in zip(y, k3)), **
        f_options)
    return tuple((k1_ + 2 * k2_ + 2 * k3_ + k4_) * (dt / 6) for k1_, k2_,
        k3_, k4_ in zip(k1, k2, k3, k4))
