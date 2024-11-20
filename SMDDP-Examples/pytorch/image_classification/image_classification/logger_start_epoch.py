def start_epoch(self):
    self.epoch += 1
    self.iteration = 0
    self.val_iteration = 0
    for n, m in self.metrics.items():
        if not n.startswith('calib'):
            m['meter'].reset_epoch()
