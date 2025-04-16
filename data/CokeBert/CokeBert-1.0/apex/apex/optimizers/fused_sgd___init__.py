def __init__(self, params, lr=required, momentum=0, dampening=0,
    weight_decay=0, nesterov=False, wd_after_momentum=False,
    materialize_master_grads=True):
    if lr is not required and lr < 0.0:
        raise ValueError('Invalid learning rate: {}'.format(lr))
    if momentum < 0.0:
        raise ValueError('Invalid momentum value: {}'.format(momentum))
    if weight_decay < 0.0:
        raise ValueError('Invalid weight_decay value: {}'.format(weight_decay))
    defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
        weight_decay=weight_decay, nesterov=nesterov)
    if nesterov and (momentum <= 0 or dampening != 0):
        raise ValueError(
            'Nesterov momentum requires a momentum and zero dampening')
    super(FusedSGD, self).__init__(params, defaults)
    self.wd_after_momentum = wd_after_momentum
    self.materialize_master_grads = materialize_master_grads
    self.most_recent_scale = 1.0
    self.scale_set_by_backward = False
    if multi_tensor_applier.available:
        import amp_C
        self._dummy_overflow_buf = torch.cuda.IntTensor([0])
        self.multi_tensor_sgd = amp_C.multi_tensor_sgd
    else:
        raise RuntimeError('apex.optimizers.FusedSGD requires cuda extensions')
