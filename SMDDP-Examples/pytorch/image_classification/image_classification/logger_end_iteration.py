def end_iteration(self, mode='train'):
    if mode == 'val':
        it = self.val_iteration
    elif mode == 'train':
        it = self.iteration
    elif mode == 'calib':
        it = self.calib_iteration
    if it % self.print_interval == 0 or mode == 'calib':
        metrics = {n: m for n, m in self.metrics.items() if n.startswith(mode)}
        if mode == 'train':
            step = self.epoch, self.iteration
        elif mode == 'val':
            step = self.epoch, self.iteration, self.val_iteration
        elif mode == 'calib':
            step = 'Calibration', self.calib_iteration
        verbositys = {m['level'] for _, m in metrics.items()}
        for ll in verbositys:
            llm = {n: m for n, m in metrics.items() if m['level'] == ll}
            dllogger.log(step=step, data={n: m['meter'].get_iteration() for
                n, m in llm.items()}, verbosity=ll)
        for n, m in metrics.items():
            m['meter'].reset_iteration()
        dllogger.flush()
