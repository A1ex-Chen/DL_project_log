def lazy_init_no_master_weights(self):
    stash = self._amp_stash
    stash.all_fp16_params = []
    stash.all_fp32_params = []
    for i, param_group in enumerate(self.param_groups):
        for i, param in enumerate(param_group['params']):
            if param.type() == 'torch.cuda.HalfTensor':
                stash.all_fp16_params.append(param)
            elif param.type() == 'torch.cuda.FloatTensor':
                stash.all_fp32_params.append(param)
            else:
                raise TypeError(
                    "Optimizer's parameters must be either torch.cuda.FloatTensor or torch.cuda.HalfTensor. Received {}"
                    .format(param.type()))
    stash.all_fp16_grad_stash = [None for _ in stash.all_fp16_params]
    stash.all_fp32_grad_stash = [None for _ in stash.all_fp32_params]
