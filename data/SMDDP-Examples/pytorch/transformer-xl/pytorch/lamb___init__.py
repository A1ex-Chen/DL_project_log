def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-06,
    weight_decay=0, adam=False):
    if not 0.0 <= lr:
        raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= eps:
        raise ValueError('Invalid epsilon value: {}'.format(eps))
    if not 0.0 <= betas[0] < 1.0:
        raise ValueError('Invalid beta parameter at index 0: {}'.format(
            betas[0]))
    if not 0.0 <= betas[1] < 1.0:
        raise ValueError('Invalid beta parameter at index 1: {}'.format(
            betas[1]))
    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    self.adam = adam
    super().__init__(params, defaults)
