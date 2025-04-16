def __init__(self, params, lr=0.001, betas=(0.95, 0), eps=1e-08,
    weight_decay=0, grad_averaging=False, amsgrad=False):
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
    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
        grad_averaging=grad_averaging, amsgrad=amsgrad)
    super(Novograd, self).__init__(params, defaults)
