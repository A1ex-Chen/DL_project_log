def _cur_loss_scaler(self):
    assert 0 <= self._loss_idx < self._num_loss
    return self._loss_scaler[self._loss_idx]
