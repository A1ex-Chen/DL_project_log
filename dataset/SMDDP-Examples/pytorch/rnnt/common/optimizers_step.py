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
                raise RuntimeError('Sparse gradients are not supported.')
            amsgrad = group['amsgrad']
            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros([]).to(state['exp_avg'].
                    device)
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros([]).to(state[
                        'exp_avg'].device)
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            if amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
            beta1, beta2 = group['betas']
            state['step'] += 1
            norm = torch.sum(torch.pow(grad, 2))
            if exp_avg_sq == 0:
                exp_avg_sq.copy_(norm)
            else:
                exp_avg_sq.mul_(beta2).add_(norm, alpha=1 - beta2)
            if amsgrad:
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = max_exp_avg_sq.sqrt().add_(group['eps'])
            else:
                denom = exp_avg_sq.sqrt().add_(group['eps'])
            grad.div_(denom)
            if group['weight_decay'] != 0:
                grad.add_(p.data, alpha=group['weight_decay'])
            if group['grad_averaging']:
                grad.mul_(1 - beta1)
            exp_avg.mul_(beta1).add_(grad)
            p.data.add_(exp_avg, alpha=-group['lr'])
    return loss
