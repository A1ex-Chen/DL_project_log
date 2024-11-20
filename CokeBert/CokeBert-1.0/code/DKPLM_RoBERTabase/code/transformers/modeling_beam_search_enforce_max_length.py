def enforce_max_length(self):
    if self._step + 1 == self.max_length:
        self.is_finished.fill_(1)
