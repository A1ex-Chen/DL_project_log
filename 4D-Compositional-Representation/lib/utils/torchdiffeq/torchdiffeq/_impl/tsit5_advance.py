def advance(self, next_t):
    """Interpolate through the next time point, integrating as necessary."""
    n_steps = 0
    while next_t > self.rk_state.t1:
        assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(
            n_steps, self.max_num_steps)
        self.rk_state = self._adaptive_tsit5_step(self.rk_state)
        n_steps += 1
    return _interp_eval_tsit5(self.rk_state.t0, self.rk_state.t1, self.
        rk_state.interp_coeff, next_t)
