def get_lr(self):
    if self.last_epoch <= self.warmup_steps:
        if self.warmup_steps != 0:
            warmup_factor = math.exp(math.log(0.01) / self.warmup_steps)
        else:
            warmup_factor = 1.0
        inv_decay = warmup_factor ** (self.warmup_steps - self.last_epoch)
        lr = [(base_lr * inv_decay) for base_lr in self.base_lrs]
    elif self.last_epoch >= self.remain_steps:
        decay_iter = self.last_epoch - self.remain_steps
        num_decay_steps = decay_iter // self.decay_interval + 1
        num_decay_steps = min(num_decay_steps, self.decay_steps)
        lr = [(base_lr * self.decay_factor ** num_decay_steps) for base_lr in
            self.base_lrs]
    else:
        lr = [base_lr for base_lr in self.base_lrs]
    return lr
