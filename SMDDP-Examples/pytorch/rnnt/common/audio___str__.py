def __str__(self):
    """Return human-readable representation of segment."""
    return (
        '%s: num_samples=%d, sample_rate=%d, duration=%.2fsec, rms=%.2fdB' %
        (type(self), self.num_samples, self.sample_rate, self.duration,
        self.rms_db))
