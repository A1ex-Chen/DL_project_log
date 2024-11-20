def keys(self):
    with _IterationGuard(self):
        for wr in self.data:
            obj = wr()
            if obj is not None:
                yield obj
