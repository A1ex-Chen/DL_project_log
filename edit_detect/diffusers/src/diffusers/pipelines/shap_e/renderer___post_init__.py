def __post_init__(self):
    assert self.t0.shape == self.t1.shape == self.intersected.shape
