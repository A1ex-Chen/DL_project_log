def add_run_time(self, forward_ms, backward_ms):
    self._forward_ms += forward_ms
    if backward_ms is None:
        return
    if self._backward_ms is None:
        self._backward_ms = 0.0
    self._backward_ms += backward_ms
