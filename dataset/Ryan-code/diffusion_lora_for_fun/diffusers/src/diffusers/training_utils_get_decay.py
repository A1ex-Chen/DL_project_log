def get_decay(self, optimization_step: int) ->float:
    """
        Compute the decay factor for the exponential moving average.
        """
    step = max(0, optimization_step - self.update_after_step - 1)
    if step <= 0:
        return 0.0
    if self.use_ema_warmup:
        cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
    else:
        cur_decay_value = (1 + step) / (10 + step)
    cur_decay_value = min(cur_decay_value, self.decay)
    cur_decay_value = max(cur_decay_value, self.min_decay)
    return cur_decay_value
