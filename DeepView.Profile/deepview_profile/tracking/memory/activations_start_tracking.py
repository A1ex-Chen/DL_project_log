def start_tracking(self):
    super().start_tracking()
    self.grad_function_contexts.clear()
    self._callable_tracker.start_tracking()
