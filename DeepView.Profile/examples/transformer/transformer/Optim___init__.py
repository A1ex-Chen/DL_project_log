def __init__(self, optimizer, d_model, n_warmup_steps):
    self._optimizer = optimizer
    self.n_warmup_steps = n_warmup_steps
    self.n_current_steps = 0
    self.init_lr = np.power(d_model, -0.5)
