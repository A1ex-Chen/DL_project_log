def end_calibration(self):
    for n, m in self.metrics.items():
        if n.startswith('calib'):
            m['meter'].reset_iteration()
