def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule=
    'warmup_linear', b1=0.9, b2=0.999, e=1e-06, weight_decay=0.01,
    max_grad_norm=1.0):
    if lr is not required and lr < 0.0:
        raise ValueError('Invalid learning rate: {} - should be >= 0.0'.
            format(lr))
    if schedule not in SCHEDULES:
        raise ValueError('Invalid schedule parameter: {}'.format(schedule))
    if not 0.0 <= warmup < 1.0 and not warmup == -1:
        raise ValueError('Invalid warmup: {} - should be in [0.0, 1.0[ or -1'
            .format(warmup))
    if not 0.0 <= b1 < 1.0:
        raise ValueError('Invalid b1 parameter: {} - should be in [0.0, 1.0['
            .format(b1))
    if not 0.0 <= b2 < 1.0:
        raise ValueError('Invalid b2 parameter: {} - should be in [0.0, 1.0['
            .format(b2))
    if not e >= 0.0:
        raise ValueError('Invalid epsilon value: {} - should be >= 0.0'.
            format(e))
    defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=
        t_total, b1=b1, b2=b2, e=e, weight_decay=weight_decay,
        max_grad_norm=max_grad_norm)
    super(BertAdam, self).__init__(params, defaults)
