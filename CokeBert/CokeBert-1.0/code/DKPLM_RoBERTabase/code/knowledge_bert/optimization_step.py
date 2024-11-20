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
                state['next_m'] = torch.zeros_like(p.data)
                state['next_v'] = torch.zeros_like(p.data)
            next_m, next_v = state['next_m'], state['next_v']
            beta1, beta2 = group['b1'], group['b2']
            if group['max_grad_norm'] > 0:
                clip_grad_norm_(p, group['max_grad_norm'])
            next_m.mul_(beta1).add_(1 - beta1, grad)
            next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
            update = next_m / (next_v.sqrt() + group['e'])
            if group['weight_decay'] > 0.0:
                update += group['weight_decay'] * p.data
            if group['t_total'] != -1:
                schedule_fct = SCHEDULES[group['schedule']]
                lr_scheduled = group['lr'] * schedule_fct(state['step'] /
                    group['t_total'], group['warmup'])
            else:
                lr_scheduled = group['lr']
            update_with_lr = lr_scheduled * update
            p.data.add_(-update_with_lr)
            state['step'] += 1
    return loss
