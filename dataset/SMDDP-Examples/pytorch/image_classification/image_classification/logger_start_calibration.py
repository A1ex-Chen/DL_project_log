def start_calibration(self):
    self.calib_iteration = 0
    for n, m in self.metrics.items():
        if n.startswith('calib'):
            m['meter'].reset_epoch()
