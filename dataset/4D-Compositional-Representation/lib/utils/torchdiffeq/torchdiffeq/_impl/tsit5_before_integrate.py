def before_integrate(self, t):
    if self.first_step is None:
        first_step = _select_initial_step(self.func, t[0], self.y0, 4, self
            .rtol, self.atol).to(t)
    else:
        first_step = _convert_to_tensor(0.01, dtype=t.dtype, device=t.device)
    self.rk_state = _RungeKuttaState(self.y0, self.func(t[0].type_as(self.
        y0[0]), self.y0), t[0], t[0], first_step, tuple(map(lambda x: [x] *
        7, self.y0)))
