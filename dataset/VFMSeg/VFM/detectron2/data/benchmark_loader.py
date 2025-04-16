def loader():
    while True:
        for k in self.sampler:
            yield self.mapper(self.dataset[k])
