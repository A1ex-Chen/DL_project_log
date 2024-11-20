def get_lr(self) ->List[float]:
    if (not self.milestones or self.last_epoch < self.milestones[0]
        ) and self.last_warm_iter < self.num_iters:
        alpha = self.last_warm_iter / self.num_iters
        factor = (1 - self.factor) * alpha + self.factor
    else:
        factor = 1
    decay_power = bisect_right(self.milestones, self.last_epoch)
    return [(base_lr * factor * self.gamma ** decay_power) for base_lr in
        self.base_lrs]
