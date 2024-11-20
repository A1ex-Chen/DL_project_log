def advance(self, final_t):
    final_t = _convert_to_tensor(final_t).to(self.vcabm_state.prev_t[0])
    while final_t > self.vcabm_state.prev_t[0]:
        self.vcabm_state = self._adaptive_adams_step(self.vcabm_state, final_t)
    assert final_t == self.vcabm_state.prev_t[0]
    return self.vcabm_state.y_n
