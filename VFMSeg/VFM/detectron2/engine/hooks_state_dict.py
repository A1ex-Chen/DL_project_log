def state_dict(self):
    if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
        return self.scheduler.state_dict()
    return {}
