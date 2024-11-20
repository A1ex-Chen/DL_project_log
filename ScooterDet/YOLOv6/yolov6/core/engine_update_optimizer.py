def update_optimizer(self):
    curr_step = self.step + self.max_stepnum * self.epoch
    self.accumulate = max(1, round(64 / self.batch_size))
    if curr_step <= self.warmup_stepnum:
        self.accumulate = max(1, np.interp(curr_step, [0, self.
            warmup_stepnum], [1, 64 / self.batch_size]).round())
        for k, param in enumerate(self.optimizer.param_groups):
            warmup_bias_lr = self.cfg.solver.warmup_bias_lr if k == 2 else 0.0
            param['lr'] = np.interp(curr_step, [0, self.warmup_stepnum], [
                warmup_bias_lr, param['initial_lr'] * self.lf(self.epoch)])
            if 'momentum' in param:
                param['momentum'] = np.interp(curr_step, [0, self.
                    warmup_stepnum], [self.cfg.solver.warmup_momentum, self
                    .cfg.solver.momentum])
    if curr_step - self.last_opt_step >= self.accumulate:
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
        self.last_opt_step = curr_step
