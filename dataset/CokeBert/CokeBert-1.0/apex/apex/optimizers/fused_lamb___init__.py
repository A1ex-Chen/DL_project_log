def __init__(self, params, lr=0.001, bias_correction=True, betas=(0.9, 
    0.999), eps=1e-06, weight_decay=0.01, amsgrad=False, adam_w_mode=True,
    grad_averaging=True, set_grad_none=True, max_grad_norm=1.0):
    if amsgrad:
        raise RuntimeError('FusedLAMB does not support the AMSGrad variant.')
    defaults = dict(lr=lr, bias_correction=bias_correction, betas=betas,
        eps=eps, weight_decay=weight_decay, grad_averaging=grad_averaging,
        max_grad_norm=max_grad_norm)
    super(FusedLAMB, self).__init__(params, defaults)
    if multi_tensor_applier.available:
        import amp_C
        self._dummy_overflow_buf = torch.cuda.IntTensor([0])
        self.multi_tensor_lamb = amp_C.multi_tensor_lamb
    else:
        raise RuntimeError('apex.optimizers.FusedLAMB requires cuda extensions'
            )
    self.adam_w_mode = 1 if adam_w_mode else 0
    self.set_grad_none = set_grad_none
