def __init__(self, init_optimizer, static_loss_scale=1.0,
    dynamic_loss_scale=False, dynamic_loss_args=None, verbose=True):
    print(
        """
fp16_optimizer is designed to only work with apex.optimizers, and will be removed in future"""
        )
    print('To update, use updated optimizers with AMP.')
    if not torch.cuda.is_available:
        raise SystemError('Cannot use fp16 without CUDA.')
    self.optimizer = init_optimizer
    self.fp16_groups = []
    self.fp16_groups_flat = []
    self.fp32_groups_flat = []
    for i, param_group in enumerate(self.optimizer.param_groups):
        self.fp16_groups.append(param_group['params'])
        self.fp16_groups_flat.append(_flatten_dense_tensors([p.clone().
            detach() for p in self.fp16_groups[i]]))
        updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i],
            self.fp16_groups[i])
        for p, q in zip(self.fp16_groups[i], updated_params):
            p.data = q.data
        self.fp32_groups_flat.append(self.fp16_groups_flat[i].clone().float
            ().detach())
        self.fp32_groups_flat[i].requires_grad = True
        param_group['params'] = [self.fp32_groups_flat[i]]
    if dynamic_loss_scale:
        if dynamic_loss_args is not None:
            raise SystemError('Do not support dynamic loss scale args for now.'
                )
        self.dynamic_loss_scale = True
        self.cur_scale = 2 ** 16
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = 2
        self.scale_window = 1000
    else:
        self.dynamic_loss_scale = False
        self.cur_iter = 0
        self.cur_scale = static_loss_scale
    self.verbose = verbose
