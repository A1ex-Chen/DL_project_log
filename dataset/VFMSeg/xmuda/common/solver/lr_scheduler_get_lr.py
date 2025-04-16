def get_lr(self):
    return [max(self.min_lr, lr) for lr in self.scheduler.get_lr()]
