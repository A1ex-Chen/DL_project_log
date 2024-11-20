def __init__(self, optimizer, total_steps, warmup_learning_rate,
    warmup_steps, last_step=-1):
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to warmup_steps.'
            )
    self._total_steps = total_steps
    self._warmup_learning_rate = warmup_learning_rate
    self._warmup_steps = warmup_steps
    super().__init__(optimizer, last_step)
