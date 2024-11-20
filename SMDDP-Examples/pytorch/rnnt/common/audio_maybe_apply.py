def maybe_apply(self, segment, sample_rate=None):
    if self._rng.random() < self.p:
        self(segment, sample_rate)
