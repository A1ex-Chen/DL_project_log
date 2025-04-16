@contextlib.contextmanager
def scale_loss(self, loss):
    if not self._amp_handle.is_active():
        yield loss
        return
    cached_grads = []
    if self._loss_idx > 0:
        for p in master_params(self._optimizer):
            if p.grad is not None:
                cached_grads.append(p.grad.data.detach().clone())
            else:
                cached_grads.append(None)
        self._optimizer.zero_grad()
    loss_scale = self._cur_loss_scaler().loss_scale()
    yield loss * loss_scale
    self._cur_loss_scaler().clear_overflow_state()
    self._cur_loss_scaler().unscale(master_params(self._optimizer),
        master_params(self._optimizer), loss_scale)
    self._skip_next[self._loss_idx] = self._cur_loss_scaler().update_scale()
    self._loss_idx += 1
    if len(cached_grads) > 0:
        for p, cached_grad in zip(master_params(self._optimizer), cached_grads
            ):
            if cached_grad is not None:
                p.grad.data.add_(cached_grad)
        cached_grads = []
