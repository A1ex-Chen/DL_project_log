def cleanup(self):
    """Remove most recent n=staleness observations"""
    if self.staleness > 0:
        self._observations = {k: v[:-self.staleness] for k, v in self.
            _observations.items()}
        self.steps_alive -= self.staleness
