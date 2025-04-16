def _update_scale(self, skip):
    if self.dynamic_loss_scale:
        if skip:
            if self.verbose:
                print('\nGrad overflow on iteration', self.cur_iter)
                print('Using dynamic loss scale of', self.cur_scale)
            self.cur_scale = max(self.cur_scale / self.scale_factor, 1)
            self.last_overflow_iter = self.cur_iter
        elif (self.cur_iter - self.last_overflow_iter
            ) % self.scale_window == 0:
            self.cur_scale *= self.scale_factor
    elif skip:
        print('\nGrad overflow on iteration', self.cur_iter)
        print('Using static loss scale of', self.cur_scale)
    self.cur_iter += 1
    return
