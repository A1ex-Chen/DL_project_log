def update_scale(self):
    if LossScaler.has_fused_kernel and self.dynamic and not self._has_overflow:
        self._has_overflow = self._overflow_buf.item()
    if self._has_overflow and self.dynamic:
        should_skip = True
        if self._min_loss_scale:
            self._loss_scale = max(self._min_loss_scale, self._loss_scale / 2.0
                )
        else:
            self._loss_scale = self._loss_scale / 2.0
        self._unskipped = 0
    else:
        should_skip = False
        self._unskipped += 1
    if self._unskipped == self._scale_seq_len and self.dynamic:
        self._loss_scale = min(self._max_loss_scale, self._loss_scale * 2.0)
        self._unskipped = 0
    return should_skip
