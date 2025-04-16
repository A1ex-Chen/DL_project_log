def __init__(self, optimizer, scale=None, auto_scale=True, inc_factor=2.0,
    dec_factor=0.5, num_iters_be_stable=500):
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError('must provide a torch.optim.Optimizer')
    self.optimizer = optimizer
    if hasattr(self.optimizer, 'name'):
        self.name = self.optimizer.name
    param_groups_copy = []
    for i, group in enumerate(optimizer.param_groups):
        group_copy = {n: v for n, v in group.items() if n != 'params'}
        group_copy['params'] = param_fp32_copy(group['params'])
        param_groups_copy.append(group_copy)
    self.param_groups = optimizer.param_groups
    optimizer.param_groups = param_groups_copy
    self.grad_scale = scale
    self.auto_scale = auto_scale
    self.inc_factor = inc_factor
    self.dec_factor = dec_factor
    self.stable_iter_count = 0
    self.num_iters_be_stable = num_iters_be_stable
