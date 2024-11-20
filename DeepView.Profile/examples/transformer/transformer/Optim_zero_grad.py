def zero_grad(self):
    """Zero out the gradients by the inner optimizer"""
    self._optimizer.zero_grad()
