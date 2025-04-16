def __iter__(self):
    batch = []
    i = 0
    for idx in self.data_iterator(self.sampler, wrap_around=False):
        batch.append(idx)
        if len(batch) == self.batch_size:
            tbatch = self._batch(batch)
            if i >= self.start_iter * self.effective_batch_size:
                yield tbatch
                self.start_iter = 0
            i += len(batch)
            batch = []
    batch_len = len(batch)
    if batch_len > 0 and not self.drop_last:
        if self.wrap_last:
            self.sampler.wrap_around -= self.batch_size
            self.wrap_around += len(batch)
            self.wrap_around %= self.batch_size
        yield self._batch(batch)
    if self.wrap_last:
        self.sampler.wrap_around += self.batch_size
