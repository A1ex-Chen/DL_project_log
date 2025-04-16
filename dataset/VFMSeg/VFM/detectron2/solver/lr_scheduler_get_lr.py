def get_lr(self) ->List[float]:
    warmup_factor = _get_warmup_factor_at_iter(self.warmup_method, self.
        last_epoch, self.warmup_iters, self.warmup_factor)
    return [(base_lr * warmup_factor * 0.5 * (1.0 + math.cos(math.pi * self
        .last_epoch / self.max_iters))) for base_lr in self.base_lrs]
