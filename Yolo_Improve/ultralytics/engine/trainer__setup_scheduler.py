def _setup_scheduler(self):
    """Initialize training learning rate scheduler."""
    if self.args.cos_lr:
        self.lf = one_cycle(1, self.args.lrf, self.epochs)
    else:
        self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf
            ) + self.args.lrf
    self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=
        self.lf)
