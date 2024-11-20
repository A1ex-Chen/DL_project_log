def __init__(self, optimizer, amp_handle, num_loss):
    self._optimizer = optimizer
    self._amp_handle = amp_handle
    self._num_loss = num_loss
    self._loss_idx = 0
    self._skip_next = [False] * num_loss
    self._loss_scaler = [LossScaler('dynamic') for _ in range(num_loss)]
