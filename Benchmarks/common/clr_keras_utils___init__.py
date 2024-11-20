def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000.0, mode=
    'triangular', gamma=1.0, scale_fn=None, scale_mode='cycle'):
    super(CyclicLR, self).__init__()
    if mode not in ['triangular', 'triangular2', 'exp_range']:
        raise KeyError(
            "mode must be one of 'triangular', 'triangular2', or 'exp_range'")
    self.base_lr = base_lr
    self.max_lr = max_lr
    self.step_size = step_size
    self.mode = mode
    self.gamma = gamma
    if scale_fn is None:
        if self.mode == 'triangular':
            self.scale_fn = lambda x: 1.0
            self.scale_mode = 'cycle'
        elif self.mode == 'triangular2':
            self.scale_fn = lambda x: 1 / 2.0 ** (x - 1)
            self.scale_mode = 'cycle'
        elif self.mode == 'exp_range':
            self.scale_fn = lambda x: gamma ** x
            self.scale_mode = 'iterations'
    else:
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
    self.clr_iterations = 0.0
    self.trn_iterations = 0.0
    self.history = {}
    self._reset()
