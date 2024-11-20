def items(self):
    with _IterationGuard(self):
        for wr, value in self.data.items():
            key = wr()
            if key is not None:
                yield key, value
