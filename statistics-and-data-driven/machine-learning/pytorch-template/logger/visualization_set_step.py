def set_step(self, step, mode='train'):
    self.mode = mode
    self.step = step
    if step == 0:
        self.timer = datetime.now()
    else:
        duration = datetime.now() - self.timer
        self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
        self.timer = datetime.now()
