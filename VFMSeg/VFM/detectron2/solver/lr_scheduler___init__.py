def __init__(self, optimizer: torch.optim.Optimizer, max_iters: int,
    warmup_factor: float=0.001, warmup_iters: int=1000, warmup_method: str=
    'linear', last_epoch: int=-1):
    logger.warning(
        'WarmupCosineLR is deprecated! Use LRMultipilier with fvcore ParamScheduler instead!'
        )
    self.max_iters = max_iters
    self.warmup_factor = warmup_factor
    self.warmup_iters = warmup_iters
    self.warmup_method = warmup_method
    super().__init__(optimizer, last_epoch)
