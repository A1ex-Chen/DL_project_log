def __init__(self, params, lr=0.001, bias_correction=True, betas=(0.9, 
    0.999), eps=1e-08, weight_decay=0.0, amsgrad=False, reg_inside_moment=
    False, grad_averaging=True, norm_type=2, init_zero=False, set_grad_none
    =True):
    if amsgrad:
        raise RuntimeError(
            'FusedNovoGrad does not support the AMSGrad variant.')
    defaults = dict(lr=lr, bias_correction=bias_correction, betas=betas,
        eps=eps, weight_decay=weight_decay, grad_averaging=grad_averaging,
        norm_type=norm_type, init_zero=init_zero)
    super(FusedNovoGrad, self).__init__(params, defaults)
    if multi_tensor_applier.available:
        import amp_C
        self._dummy_overflow_buf = torch.cuda.IntTensor([0])
        self.multi_tensor_novograd = amp_C.multi_tensor_novograd
    else:
        raise RuntimeError(
            'apex.optimizers.FusedNovoGrad requires cuda extensions')
    self.moment_mode = 0 if reg_inside_moment else 1
    self.set_grad_none = set_grad_none
