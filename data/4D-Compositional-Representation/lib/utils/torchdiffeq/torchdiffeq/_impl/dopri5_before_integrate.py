def before_integrate(self, t):
    f0 = self.func(t[0].type_as(self.y0[0]), self.y0, **self.f_options)
    if self.first_step is None:
        first_step = _select_initial_step(self.func, t[0], self.y0, self.
            f_options, 4, self.rtol[0], self.atol[0], f0=f0).to(t)
    else:
        first_step = _convert_to_tensor(0.01, dtype=t.dtype, device=t.device)
    self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], first_step,
        interp_coeff=[self.y0] * 5)
