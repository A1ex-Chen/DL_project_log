def update(self, key, value, n=1):
    if self.writer is not None:
        self.writer.add_scalar(key, value)
    self._data.total[key] += value * n
    self._data.counts[key] += n
    self._data.average[key] = self._data.total[key] / self._data.counts[key]
