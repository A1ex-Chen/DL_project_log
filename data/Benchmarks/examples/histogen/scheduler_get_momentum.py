def get_momentum(self):
    if self.iteration > 2 * self.cycle_step:
        momentum = self.momentum[0]
    elif self.iteration > self.cycle_step:
        cut = 1 - (self.iteration - self.cycle_step) / self.cycle_step
        momentum = self.momentum[0] + cut * (self.momentum[1] - self.
            momentum[0])
    else:
        cut = self.iteration / self.cycle_step
        momentum = self.momentum[0] + cut * (self.momentum[1] - self.
            momentum[0])
    return momentum
