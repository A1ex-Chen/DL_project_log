def new_add_param_group(self, new_group):
    stash = self._amp_stash
    if not stash.lazy_init_called:
        self._lazy_init_maybe_master_weights()
        stash.lazy_init_called = True
    assert isinstance(new_group, dict), 'param group must be a dict'
    new_params = new_group['params']
    if isinstance(new_params, torch.Tensor):
        new_group['params'] = [new_params]
    elif isinstance(new_params, set):
        raise TypeError(
            'optimizer parameters need to be organized in ordered collections, but the ordering of tensors in sets will change between runs. Please use a list instead.'
            )
    else:
        new_group['params'] = list(new_params)
    if properties.master_weights:
        fp16_params_this_group = []
        fp32_params_this_group = []
        fp32_from_fp16_params_this_group = []
        for i, param in enumerate(new_group['params']):
            if param.requires_grad:
                if param.type() == 'torch.cuda.HalfTensor':
                    fp16_params_this_group.append(param)
                    master_param = param.detach().clone().float()
                    master_param.requires_grad = True
                    new_group['params'][i] = master_param
                    fp32_from_fp16_params_this_group.append(master_param)
                elif param.type() == 'torch.cuda.FloatTensor':
                    fp32_params_this_group.append(param)
                    new_group['params'][i] = param
                else:
                    raise TypeError(
                        "Optimizer's parameters must be either torch.cuda.FloatTensor or torch.cuda.HalfTensor. Received {}"
                        .format(param.type()))
        stash.fp16_groups.append(fp16_params_this_group)
        stash.fp32_from_fp16_groups.append(fp32_from_fp16_params_this_group)
        stash.fp32_from_fp32_groups.append(fp32_params_this_group)
        stash.all_fp16_params += fp16_params_this_group
        stash.all_fp32_from_fp16_params += fp32_from_fp16_params_this_group
        stash.all_fp32_from_fp32_params += fp32_params_this_group
        stash.all_fp32_from_fp32_grad_stash += [None for _ in
            fp32_params_this_group]
    else:
        for param in new_group['params']:
            if param.type() == 'torch.cuda.HalfTensor':
                stash.all_fp16_params.append(param)
                stash.all_fp16_grad_stash.append(None)
            elif param.type() == 'torch.cuda.FloatTensor':
                stash.all_fp32_params.append(param)
                stash.all_fp32_grad_stash.append(None)
            else:
                raise TypeError(
                    "Optimizer's parameters must be either torch.cuda.FloatTensor or torch.cuda.HalfTensor. Received {}"
                    .format(param.type()))
    old_add_param_group(new_group)
