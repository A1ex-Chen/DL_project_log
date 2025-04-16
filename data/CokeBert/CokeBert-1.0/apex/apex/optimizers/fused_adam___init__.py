def __init__(self, params, lr=0.001, bias_correction=True, betas=(0.9, 
    0.999), eps=1e-08, eps_inside_sqrt=False, weight_decay=0.0,
    max_grad_norm=0.0, amsgrad=False):
    global fused_adam_cuda
    fused_adam_cuda = importlib.import_module('fused_adam_cuda')
    if amsgrad:
        raise RuntimeError('FusedAdam does not support the AMSGrad variant.')
    defaults = dict(lr=lr, bias_correction=bias_correction, betas=betas,
        eps=eps, weight_decay=weight_decay, max_grad_norm=max_grad_norm)
    super(FusedAdam, self).__init__(params, defaults)
    self.eps_mode = 0 if eps_inside_sqrt else 1
