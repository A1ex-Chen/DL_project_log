def flow(self, X, y, batch_size=32, shuffle=False, seed=None):
    assert len(X) == len(y)
    self.X = X
    self.y = y
    self.flow_generator = self._flow_index(X.shape[0], batch_size, shuffle,
        seed)
    return self
