def __post_init__(self):
    assert self.x.shape[0] == self.y.shape[0] == self.z.shape[0
        ] == self.origin.shape[0]
    assert self.x.shape[1] == self.y.shape[1] == self.z.shape[1
        ] == self.origin.shape[1] == 3
    assert len(self.x.shape) == len(self.y.shape) == len(self.z.shape) == len(
        self.origin.shape) == 2
