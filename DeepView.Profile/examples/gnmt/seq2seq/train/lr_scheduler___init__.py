def __init__(self, optimizer, iterations, warmup_steps=0, remain_steps=1.0,
    decay_interval=None, decay_steps=4, decay_factor=0.5, last_epoch=-1):
    """
        Constructor of WarmupMultiStepLR.

        Parameters: warmup_steps, remain_steps and decay_interval accept both
        integers and floats as an input. Integer input is interpreted as
        absolute index of iteration, float input is interpreted as a fraction
        of total training iterations (epochs * steps_per_epoch).

        If decay_interval is None then the decay will happen at regulary spaced
        intervals ('decay_steps' decays between iteration indices
        'remain_steps' and 'iterations').

        :param optimizer: instance of optimizer
        :param iterations: total number of training iterations
        :param warmup_steps: number of warmup iterations
        :param remain_steps: start decay at 'remain_steps' iteration
        :param decay_interval: interval between LR decay steps
        :param decay_steps: max number of decay steps
        :param decay_factor: decay factor
        :param last_epoch: the index of last iteration
        """
    self.warmup_steps = perhaps_convert_float(warmup_steps, iterations)
    self.remain_steps = perhaps_convert_float(remain_steps, iterations)
    if decay_interval is None:
        decay_iterations = iterations - self.remain_steps
        self.decay_interval = decay_iterations // decay_steps
        self.decay_interval = max(self.decay_interval, 1)
    else:
        self.decay_interval = perhaps_convert_float(decay_interval, iterations)
    self.decay_factor = decay_factor
    self.decay_steps = decay_steps
    if self.warmup_steps > self.remain_steps:
        logging.warn(
            'warmup_steps should not be larger than remain_steps, setting warmup_steps=remain_steps'
            )
        self.warmup_steps = self.remain_steps
    super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)
