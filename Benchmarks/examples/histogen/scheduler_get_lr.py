def get_lr(self):
    lr = (self.lr_mult * self.iteration if self.linear else self.lr_mult **
        self.iteration)
    lr = self.lr_min + lr if self.linear else self.lr_min * lr
    self.iteration += 1
    self.lrs.append(lr)
    return [lr for base_lr in self.base_lrs]
