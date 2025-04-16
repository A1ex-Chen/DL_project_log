def __init__(self, optimizer, max_epoch, min_lr, init_lr, warmup_steps=0,
    warmup_start_lr=-1, **kwargs):
    self.optimizer = optimizer
    self.max_epoch = max_epoch
    self.min_lr = min_lr
    self.init_lr = init_lr
    self.warmup_steps = warmup_steps
    self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
