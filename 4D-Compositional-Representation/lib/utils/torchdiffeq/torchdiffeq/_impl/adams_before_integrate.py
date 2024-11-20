def before_integrate(self, t):
    prev_f = collections.deque(maxlen=self.max_order + 1)
    prev_t = collections.deque(maxlen=self.max_order + 1)
    phi = collections.deque(maxlen=self.max_order)
    t0 = t[0]
    f0 = self.func(t0.type_as(self.y0[0]), self.y0, **self.f_options)
    prev_t.appendleft(t0)
    prev_f.appendleft(f0)
    phi.appendleft(f0)
    first_step = _select_initial_step(self.func, t[0], self.y0, self.
        f_options, 2, self.rtol[0], self.atol[0], f0=f0).to(t)
    self.vcabm_state = _VCABMState(self.y0, prev_f, prev_t, next_t=t[0] +
        first_step, phi=phi, order=1)
