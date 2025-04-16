def get_lr(self):
    ret = [self._get_lr_per_group(base_lr) for base_lr in self.base_lrs]
    return ret
