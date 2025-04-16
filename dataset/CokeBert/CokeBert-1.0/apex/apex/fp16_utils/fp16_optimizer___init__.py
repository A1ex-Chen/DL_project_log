def __init__(self, init_optimizer, static_loss_scale=1.0,
    dynamic_loss_scale=False, dynamic_loss_args=None, verbose=True):
    if not torch.cuda.is_available:
        raise SystemError('Cannot use fp16 without CUDA.')
    self.verbose = verbose
    self.optimizer = init_optimizer
    self.fp16_groups = []
    self.fp32_from_fp16_groups = []
    self.fp32_from_fp32_groups = []
    for i, param_group in enumerate(self.optimizer.param_groups):
        self.maybe_print('FP16_Optimizer processing param group {}:'.format(i))
        fp16_params_this_group = []
        fp32_params_this_group = []
        fp32_from_fp16_params_this_group = []
        for i, param in enumerate(param_group['params']):
            if param.requires_grad:
                if param.type() == 'torch.cuda.HalfTensor':
                    self.maybe_print(
                        'FP16_Optimizer received torch.cuda.HalfTensor with {}'
                        .format(param.size()))
                    fp16_params_this_group.append(param)
                    master_param = param.detach().clone().float()
                    master_param.requires_grad = True
                    param_group['params'][i] = master_param
                    fp32_from_fp16_params_this_group.append(master_param)
                    if param in self.optimizer.state:
                        self.optimizer.state[master_param
                            ] = self.optimizer.state.pop(param)
                elif param.type() == 'torch.cuda.FloatTensor':
                    self.maybe_print(
                        'FP16_Optimizer received torch.cuda.FloatTensor with {}'
                        .format(param.size()))
                    fp32_params_this_group.append(param)
                    param_group['params'][i] = param
                else:
                    raise TypeError(
                        'Wrapped parameters must be either torch.cuda.FloatTensor or torch.cuda.HalfTensor. Received {}'
                        .format(param.type()))
        self.fp16_groups.append(fp16_params_this_group)
        self.fp32_from_fp16_groups.append(fp32_from_fp16_params_this_group)
        self.fp32_from_fp32_groups.append(fp32_params_this_group)
    self.all_fp16_params = []
    for group in self.fp16_groups:
        self.all_fp16_params += group
    self.all_fp32_from_fp16_params = []
    for group in self.fp32_from_fp16_groups:
        self.all_fp32_from_fp16_params += group
    self.all_fp32_from_fp32_params = []
    for group in self.fp32_from_fp32_groups:
        self.all_fp32_from_fp32_params += group
    self.optimizer.load_state_dict(self.optimizer.state_dict())
    if dynamic_loss_scale:
        self.dynamic_loss_scale = True
        if dynamic_loss_args is not None:
            self.loss_scaler = LossScaler('dynamic', **dynamic_loss_args)
        else:
            self.loss_scaler = LossScaler('dynamic')
    else:
        self.dynamic_loss_scale = False
        self.loss_scaler = LossScaler(static_loss_scale)
    self.overflow = False
    self.first_closure_call_this_step = True
    self.clip_grad_norm = clip_grad_norm
    if multi_tensor_applier.available:
        import amp_C
        self.multi_tensor_scale = amp_C.multi_tensor_scale
        self._dummy_overflow_buf = torch.cuda.IntTensor([0])
