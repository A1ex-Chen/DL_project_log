def zero_grad(self):
    if self.set_grad_none:
        for group in self.param_groups:
            for p in group['params']:
                p.grad = None
    else:
        super(FusedLAMB, self).zero_grad()
