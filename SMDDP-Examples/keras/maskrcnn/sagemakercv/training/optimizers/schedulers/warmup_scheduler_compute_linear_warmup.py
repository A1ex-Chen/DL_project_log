def compute_linear_warmup(self, step):
    return (self.scheduler_learning_rate * step + self.
        initial_learning_rate * (self.warmup_steps - step)) / self.warmup_steps
