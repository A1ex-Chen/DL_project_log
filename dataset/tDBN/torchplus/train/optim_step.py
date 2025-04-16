def step(self, closure=None):
    for g, g_copy in zip(self.param_groups, self.optimizer.param_groups):
        invalid = set_grad(g_copy['params'], g['params'], self.grad_scale)
        if invalid:
            if self.grad_scale is None or self.auto_scale is False:
                raise ValueError('nan/inf detected but auto_scale disabled.')
            self.grad_scale *= self.dec_factor
            print('scale decay to {}'.format(self.grad_scale))
            return
    if self.auto_scale is True:
        self.stable_iter_count += 1
        if self.stable_iter_count > self.num_iters_be_stable:
            if self.grad_scale is not None:
                self.grad_scale *= self.inc_factor
            self.stable_iter_count = 0
    if closure is None:
        self.optimizer.step()
    else:
        self.optimizer.step(closure)
    for g, g_copy in zip(self.param_groups, self.optimizer.param_groups):
        for p_copy, p in zip(g_copy['params'], g['params']):
            p.data.copy_(p_copy.data)
