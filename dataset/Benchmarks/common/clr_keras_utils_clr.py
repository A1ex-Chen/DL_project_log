def clr(self):
    cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
    x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
    if self.scale_mode == 'cycle':
        return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, 
            1 - x) * self.scale_fn(cycle)
    else:
        return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, 
            1 - x) * self.scale_fn(self.clr_iterations)
