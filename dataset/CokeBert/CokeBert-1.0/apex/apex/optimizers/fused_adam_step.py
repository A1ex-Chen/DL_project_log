def step(self, closure=None, grads=None, output_params=None, scale=1.0,
    grad_norms=None):
    """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output params (list of tensors, optional): A reduced precision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients. (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
    loss = None
    if closure is not None:
        loss = closure()
    if grads is None:
        grads_group = [None] * len(self.param_groups)
    elif isinstance(grads, types.GeneratorType):
        grads_group = [grads]
    elif type(grads[0]) != list:
        grads_group = [grads]
    else:
        grads_group = grads
    if output_params is None:
        output_params_group = [None] * len(self.param_groups)
    elif isinstance(output_params, types.GeneratorType):
        output_params_group = [output_params]
    elif type(output_params[0]) != list:
        output_params_group = [output_params]
    else:
        output_params_group = output_params
    if grad_norms is None:
        grad_norms = [None] * len(self.param_groups)
    for group, grads_this_group, output_params_this_group, grad_norm in zip(
        self.param_groups, grads_group, output_params_group, grad_norms):
        if grads_this_group is None:
            grads_this_group = [None] * len(group['params'])
        if output_params_this_group is None:
            output_params_this_group = [None] * len(group['params'])
        combined_scale = scale
        if group['max_grad_norm'] > 0:
            clip = (grad_norm / scale + 1e-06) / group['max_grad_norm']
            if clip > 1:
                combined_scale = clip * scale
        bias_correction = 1 if group['bias_correction'] else 0
        for p, grad, output_param in zip(group['params'], grads_this_group,
            output_params_this_group):
            if p.grad is None and grad is None:
                continue
            if grad is None:
                grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError(
                    'FusedAdam does not support sparse gradients, please consider SparseAdam instead'
                    )
            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']
            state['step'] += 1
            out_p = torch.tensor([], dtype=torch.float
                ) if output_param is None else output_param
            fused_adam_cuda.adam(p.data, out_p, exp_avg, exp_avg_sq, grad,
                group['lr'], beta1, beta2, group['eps'], combined_scale,
                state['step'], self.eps_mode, bias_correction, group[
                'weight_decay'])
    return loss
