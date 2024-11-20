def _get_lr_per_group(self, base_lr):
    if base_lr < self._warmup_learning_rate:
        raise ValueError(
            'learning_rate_base must be larger or equal to warmup_learning_rate.'
            )
    step = self.last_step
    learning_rate = 0.5 * base_lr * (1 + np.cos(np.pi * (float(step) - self
        ._warmup_steps) / float(self._total_steps - self._warmup_steps)))
    if self._warmup_steps > 0:
        slope = (base_lr - self._warmup_learning_rate) / self._warmup_steps
        pre_cosine_learning_rate = slope * float(step
            ) + self._warmup_learning_rate
        if step < self._warmup_steps:
            return pre_cosine_learning_rate
        else:
            return learning_rate
