def __eq__(self, other):
    """Return whether two objects are equal."""
    if type(other) is not type(self):
        return False
    if self._sample_rate != other._sample_rate:
        return False
    if self._samples.shape != other._samples.shape:
        return False
    if np.any(self.samples != other._samples):
        return False
    return True
