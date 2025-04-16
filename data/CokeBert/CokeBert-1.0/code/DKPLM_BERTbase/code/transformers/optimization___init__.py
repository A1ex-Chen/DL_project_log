def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-06,
    weight_decay=0.0, correct_bias=True):
    if lr < 0.0:
        raise ValueError('Invalid learning rate: {} - should be >= 0.0'.
            format(lr))
    if not 0.0 <= betas[0] < 1.0:
        raise ValueError('Invalid beta parameter: {} - should be in [0.0, 1.0['
            .format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
        raise ValueError('Invalid beta parameter: {} - should be in [0.0, 1.0['
            .format(betas[1]))
    if not 0.0 <= eps:
        raise ValueError('Invalid epsilon value: {} - should be >= 0.0'.
            format(eps))
    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
        correct_bias=correct_bias)
    super(AdamW, self).__init__(params, defaults)
