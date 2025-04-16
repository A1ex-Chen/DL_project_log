def lazy_init_with_master_weights(self):
    stash = self._amp_stash
    stash.fp16_groups = []
    stash.fp32_from_fp16_groups = []
    stash.fp32_from_fp32_groups = []
    for i, param_group in enumerate(self.param_groups):
        fp16_params_this_group = []
        fp32_params_this_group = []
        fp32_from_fp16_params_this_group = []
        for i, param in enumerate(param_group['params']):
            if param.requires_grad:
                if param.type() == 'torch.cuda.HalfTensor':
                    fp16_params_this_group.append(param)
                    master_param = param.detach().clone().float()
                    master_param.requires_grad = True
                    param_group['params'][i] = master_param
                    fp32_from_fp16_params_this_group.append(master_param)
                    if param in self.state:
                        self.state[master_param] = self.state.pop(param)
                elif param.type() == 'torch.cuda.FloatTensor':
                    fp32_params_this_group.append(param)
                    param_group['params'][i] = param
                else:
                    raise TypeError(
                        "Optimizer's parameters must be either torch.cuda.FloatTensor or torch.cuda.HalfTensor. Received {}"
                        .format(param.type()))
        stash.fp16_groups.append(fp16_params_this_group)
        stash.fp32_from_fp16_groups.append(fp32_from_fp16_params_this_group)
        stash.fp32_from_fp32_groups.append(fp32_params_this_group)
    stash.all_fp16_params = []
    for group in stash.fp16_groups:
        stash.all_fp16_params += group
    stash.all_fp32_from_fp16_params = []
    for group in stash.fp32_from_fp16_groups:
        stash.all_fp32_from_fp16_params += group
    stash.all_fp32_from_fp32_params = []
    for group in stash.fp32_from_fp32_groups:
        stash.all_fp32_from_fp32_params += group
    stash.all_fp32_from_fp32_grad_stash = [None for _ in stash.
        all_fp32_from_fp32_params]
    for param in stash.all_fp32_from_fp16_params:
        param.grad = None
    for param in stash.all_fp32_from_fp32_params:
        param.grad = None
    self.load_state_dict(self.state_dict())
