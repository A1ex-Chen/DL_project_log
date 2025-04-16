def __init__(self, rate, **kwargs):
    super(PermanentDropout, self).__init__(rate, **kwargs)
    self.uses_learning_phase = False
