def train(self):
    """
        Full training logic
        """
    not_improved_count = 0
    for epoch in range(self.start_epoch, self.epochs + 1):
        result = self._train_epoch(epoch)
        log = {'epoch': epoch}
        log.update(result)
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))
        best = False
        if self.mnt_mode != 'off':
            try:
                improved = self.mnt_mode == 'min' and log[self.mnt_metric
                    ] <= self.mnt_best or self.mnt_mode == 'max' and log[self
                    .mnt_metric] >= self.mnt_best
            except KeyError:
                self.logger.warning(
                    "Warning: Metric '{}' is not found. Model performance monitoring is disabled."
                    .format(self.mnt_metric))
                self.mnt_mode = 'off'
                improved = False
            if improved:
                self.mnt_best = log[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1
            if not_improved_count > self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. Training stops."
                    .format(self.early_stop))
                break
        if epoch % self.save_period == 0:
            self._save_checkpoint(epoch, save_best=best)
