def _update_learning_rate(self):
    """ Learning rate scheduling per step """
    self.n_current_steps += 1
    lr = self.init_lr * self._get_lr_scale()
    for param_group in self._optimizer.param_groups:
        param_group['lr'] = lr
