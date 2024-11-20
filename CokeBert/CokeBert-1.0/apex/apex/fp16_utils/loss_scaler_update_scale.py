def update_scale(self, overflow):
    if overflow:
        self.cur_scale = max(self.cur_scale / self.scale_factor, 1)
        self.last_overflow_iter = self.cur_iter
    elif (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
        self.cur_scale *= self.scale_factor
    self.cur_iter += 1
