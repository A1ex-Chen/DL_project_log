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
                raise RuntimeError(
                    'Adam does not support sparse gradients, please consider SparseAdam instead'
                    )
            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']
            state['step'] += 1
            exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
            exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
            denom = exp_avg_sq.sqrt().add_(group['eps'])
            step_size = group['lr']
            if group['correct_bias']:
                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']
                step_size = step_size * math.sqrt(bias_correction2
                    ) / bias_correction1
            p.data.addcdiv_(-step_size, exp_avg, denom)
            if group['weight_decay'] > 0.0:
                p.data.add_(-group['lr'] * group['weight_decay'], p.data)
    return loss
