def apply_gradients(self, grads_and_vars, name: Text=None):
    result = self._optimizer.apply_gradients(grads_and_vars, name)
    self.update_average(self._optimizer.iterations)
    return result
