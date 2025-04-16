def step_func(self, func, t, dt, y, f_options):
    return rk_common.rk4_alt_step_func(func, t, dt, y, f_options)
