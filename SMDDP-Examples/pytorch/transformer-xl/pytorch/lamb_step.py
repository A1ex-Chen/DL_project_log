def step(self, closure=None):
    """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
    loss = None
    if closure is not None:
        loss = closure()
    for group in self.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('Lamb does not support sparse gradients.')
            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']
            state['step'] += 1
            step_size = group['lr']
            param, exp_avg, exp_avg_sq = lamb_kernel(p.data, grad, exp_avg,
                exp_avg_sq, beta1, beta2, step_size, group['eps'], group[
                'weight_decay'])
            state['exp_avg'] = exp_avg
            state['exp_avg_sq'] = exp_avg_sq
            p.data = param
    return loss
