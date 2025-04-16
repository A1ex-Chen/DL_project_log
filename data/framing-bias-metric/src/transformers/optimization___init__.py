def __init__(self, params, lr=None, eps=(1e-30, 0.001), clip_threshold=1.0,
    decay_rate=-0.8, beta1=None, weight_decay=0.0, scale_parameter=True,
    relative_step=True, warmup_init=False):
    if lr is not None and relative_step:
        raise ValueError('Cannot combine manual lr and relative_step options')
    if warmup_init and not relative_step:
        raise ValueError('warmup_init requires relative_step=True')
    defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold,
        decay_rate=decay_rate, beta1=beta1, weight_decay=weight_decay,
        scale_parameter=scale_parameter, relative_step=relative_step,
        warmup_init=warmup_init)
    super().__init__(params, defaults)
