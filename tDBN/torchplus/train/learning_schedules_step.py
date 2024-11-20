def step(self, step=None):
    if step is None:
        step = self.last_step + 1
    self.last_step = step
    for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
        param_group['lr'] = lr
