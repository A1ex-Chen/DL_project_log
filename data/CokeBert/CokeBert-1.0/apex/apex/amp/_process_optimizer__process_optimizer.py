def _process_optimizer(optimizer, properties):
    if hasattr(optimizer, '_amp_stash'):
        raise RuntimeError(
            'A given optimizer should only be passed through amp.initialize once.'
            )
    else:
        optimizer._amp_stash = AmpOptimizerState()
    optimizer._amp_stash.lazy_init_called = False
    optimizer._amp_stash.already_patched = False
    optimizer._amp_stash.params_have_scaled_gradients = False
    for name in ('_lazy_init_maybe_master_weights',
        '_master_params_to_model_params', '_prepare_amp_backward',
        '_post_amp_backward'):
        if hasattr(optimizer, name):
            raise RuntimeError('Incoming optimizer already has {} defined.'
                .format(name))
    if multi_tensor_applier.available:
        import amp_C
        optimizer._amp_stash.multi_tensor_scale = amp_C.multi_tensor_scale
        optimizer._amp_stash.dummy_overflow_buf = torch.cuda.IntTensor([0])
    if properties.master_weights:
        optimizer._lazy_init_maybe_master_weights = types.MethodType(
            lazy_init_with_master_weights, optimizer)
        optimizer._master_params_to_model_params = types.MethodType(
            _master_params_to_model_params, optimizer)
        old_step = optimizer.step

        def new_step(self, closure=None):
            if closure is not None:
                raise RuntimeError(
                    'Currently, Amp does not support closure use with optimizers.'
                    )
            retval = old_step()
            self._master_params_to_model_params()
            for param in self._amp_stash.all_fp32_from_fp16_params:
                param.grad = None
            return retval
        optimizer.step = types.MethodType(new_step, optimizer)
        old_zero_grad = optimizer.zero_grad

        def new_zero_grad(self):
            stash = self._amp_stash
            if not stash.lazy_init_called:
                self._lazy_init_maybe_master_weights()
                stash.lazy_init_called = True
            for param in stash.all_fp16_params:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()
            for param in stash.all_fp32_from_fp32_params:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()
            for param in self._amp_stash.all_fp32_from_fp16_params:
                param.grad = None
        optimizer.zero_grad = types.MethodType(new_zero_grad, optimizer)
        optimizer._prepare_amp_backward = types.MethodType(
            prepare_backward_with_master_weights, optimizer)
        optimizer._post_amp_backward = types.MethodType(
            post_backward_with_master_weights, optimizer)
    else:
        optimizer._lazy_init_maybe_master_weights = types.MethodType(
            lazy_init_no_master_weights, optimizer)
        optimizer._prepare_amp_backward = types.MethodType(
            prepare_backward_no_master_weights, optimizer)
        optimizer._post_amp_backward = types.MethodType(
            post_backward_no_master_weights, optimizer)
    old_add_param_group = optimizer.add_param_group

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
            stash.fp32_from_fp16_groups.append(fp32_from_fp16_params_this_group
                )
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
    optimizer.add_param_group = types.MethodType(new_add_param_group, optimizer
        )
    return optimizer
