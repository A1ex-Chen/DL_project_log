def __init__(self, optimizer, lr_min, lr_max, step_size, linear=False):
    ratio = lr_max / lr_min
    self.linear = linear
    self.lr_min = lr_min
    self.lr_mult = ratio / step_size if linear else ratio ** (1 / step_size)
    self.iteration = 0
    self.lrs = []
    self.losses = []
    super().__init__(optimizer, -1)
