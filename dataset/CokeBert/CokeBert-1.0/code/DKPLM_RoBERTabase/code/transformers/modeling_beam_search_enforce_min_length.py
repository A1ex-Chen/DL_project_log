def enforce_min_length(self):
    if self._step < self.min_length:
        self.log_probabilities[self.end_token_id] = -1e+20
