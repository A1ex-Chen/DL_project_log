@property
def current_lr(self):
    return self.trainer.optimizer.lr(self.trainer.optimizer.iterations)
