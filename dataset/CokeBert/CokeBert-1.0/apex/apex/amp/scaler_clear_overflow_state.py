def clear_overflow_state(self):
    self._has_overflow = False
    if self.has_fused_kernel:
        self._overflow_buf.zero_()
