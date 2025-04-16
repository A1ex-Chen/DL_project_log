def __init__(self, min_rate=0.85, max_rate=1.15, discrete=False, p=0.1, rng
    =None):
    super(SpeedPerturbation, self).__init__(p, rng)
    assert 0 < min_rate < max_rate
    self.min_rate = min_rate
    self.max_rate = max_rate
    self.discrete = discrete
