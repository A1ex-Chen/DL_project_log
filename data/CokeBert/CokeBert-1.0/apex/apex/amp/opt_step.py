def step(self, closure=None):
    if not self._amp_handle.is_active():
        return self._optimizer.step(closure=closure)
    self._loss_idx = 0
    for group in self._optimizer.param_groups:
        for p in group['params']:
            self._amp_handle.remove_cache(p)
    if closure is not None:
        raise NotImplementedError(
            'The `closure` argument is unsupported by the amp ' +
            'optimizer wrapper.')
    if any(self._skip_next):
        maybe_print('Gradient overflow, skipping update')
        self._skip_next = [False] * self._num_loss
    else:
        return self._optimizer.step(closure=closure)
