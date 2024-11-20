def __init__(self, scheduler, min_lr=1e-05):
    assert isinstance(scheduler, _LRScheduler)
    self.scheduler = scheduler
    self.min_lr = min_lr
