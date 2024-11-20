def step(self, closure=None):
    """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
    loss = None
    if closure is not None:
        loss = closure()
    explicit_master_params = hasattr(self, '_amp_stash') and hasattr(self.
        _amp_stash, 'fp32_from_fp16_groups')
    for gid, group in enumerate(self.param_groups):
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        first_runs = [True, True]
        if explicit_master_params:
            stash = self._amp_stash
            fp32_params = [p for p in stash.fp32_from_fp32_groups[gid] if p
                .grad is not None]
            fp32_grads = [p.grad for p in stash.fp32_from_fp32_groups[gid] if
                p.grad is not None]
            fp32_momentums, first_runs[1] = self.get_momentums(fp32_params)
            if self.materialize_master_grads:
                fp16_model_params = [p for i, p in enumerate(stash.
                    fp16_groups[gid]) if stash.fp32_from_fp16_groups[gid][i
                    ].grad is not None]
                fp32_from_fp16_grads = [p.grad for p in stash.
                    fp32_from_fp16_groups[gid] if p.grad is not None]
                fp32_from_fp16_params = [p for p in stash.
                    fp32_from_fp16_groups[gid] if p.grad is not None]
                fp32_from_fp16_momentums, first_runs[0] = self.get_momentums(
                    fp32_from_fp16_params)
                fp16_set = [fp32_from_fp16_grads, fp32_from_fp16_params,
                    fp32_from_fp16_momentums, fp16_model_params]
            else:
                fp16_model_params = [p for p in stash.fp16_groups[gid] if p
                    .grad is not None]
                fp16_model_grads = [p.grad for p in stash.fp16_groups[gid] if
                    p.grad is not None]
                fp32_from_fp16_params = [p for i, p in enumerate(stash.
                    fp32_from_fp16_groups[gid]) if stash.fp16_groups[gid][i
                    ].grad is not None]
                fp32_from_fp16_momentums, first_runs[0] = self.get_momentums(
                    fp32_from_fp16_params)
                fp16_set = [fp16_model_grads, fp32_from_fp16_params,
                    fp32_from_fp16_momentums, fp16_model_params]
            launch_sets = [fp16_set, [fp32_grads, fp32_params, fp32_momentums]]
        else:
            fp16_params = [p for p in group['params'] if p.dtype == torch.
                float16 and p.grad is not None]
            fp16_grads = [p.grad for p in group['params'] if p.dtype ==
                torch.float16 and p.grad is not None]
            fp16_momentums, first_runs[0] = self.get_momentums(fp16_params)
            fp32_params = [p for p in group['params'] if p.dtype == torch.
                float32 and p.grad is not None]
            fp32_grads = [p.grad for p in group['params'] if p.dtype ==
                torch.float32 and p.grad is not None]
            fp32_momentums, first_runs[1] = self.get_momentums(fp32_params)
            launch_sets = [[fp16_grads, fp16_params, fp16_momentums], [
                fp32_grads, fp32_params, fp32_momentums]]
        for s, (launch_set, first_run) in enumerate(zip(launch_sets,
            first_runs)):
            assert len(launch_set[0]) == len(launch_set[1])
            assert len(launch_set[0]) == len(launch_set[2])
            if len(launch_set[0]) > 0:
                multi_tensor_applier(self.multi_tensor_sgd, self.
                    _dummy_overflow_buf, launch_set, weight_decay, momentum,
                    dampening, group['lr'], nesterov, first_run, self.
                    wd_after_momentum, 1.0 / self.most_recent_scale)
    self.most_recent_scale = 1.0
    self.scale_set_by_backward = False
    return loss
