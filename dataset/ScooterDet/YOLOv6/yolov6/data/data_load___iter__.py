def __iter__(self):
    while True:
        yield from iter(self.sampler)
