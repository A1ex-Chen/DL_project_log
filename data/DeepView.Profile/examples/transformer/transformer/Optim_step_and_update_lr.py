def step_and_update_lr(self):
    """Step with the inner optimizer"""
    self._update_learning_rate()
    self._optimizer.step()
