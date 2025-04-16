def values(self):
    with _IterationGuard(self):
        for wr, value in self.data.items():
            if wr() is not None:
                yield value
