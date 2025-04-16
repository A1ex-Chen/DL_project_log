def step(self):
    lr = self.lr_phase[self.phase].step()
    if self.momentum is not None:
        momentum = self.momentum_phase[self.phase].step()
    else:
        momentum = None
    for group in self.optimizer.param_groups:
        group['lr'] = lr
        if self.momentum is not None:
            if 'betas' in group:
                group['betas'] = momentum, group['betas'][1]
            else:
                group['momentum'] = momentum
    if self.lr_phase[self.phase].is_done:
        self.phase += 1
    if self.phase >= len(self.lr_phase):
        for phase in self.lr_phase:
            phase.reset()
        for phase in self.momentum_phase:
            phase.reset()
        self.phase = 0
    return lr, momentum
