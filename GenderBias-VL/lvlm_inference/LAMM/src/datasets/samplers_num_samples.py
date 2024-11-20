@property
def num_samples(self):
    if self._num_samples is None:
        return len(self.data_source)
    return self._num_samples
