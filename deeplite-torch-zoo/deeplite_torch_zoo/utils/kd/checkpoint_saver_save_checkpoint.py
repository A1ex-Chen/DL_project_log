def save_checkpoint(self, epoch, metric=None, metric_ema=None):
    assert epoch >= 0
    tmp_save_path = os.path.join(self.checkpoint_dir, 'tmp' + self.extension)
    last_save_path = os.path.join(self.checkpoint_dir, 'last' + self.extension)
    self._save(tmp_save_path, epoch, metric, metric_ema)
    if os.path.exists(last_save_path):
        os.unlink(last_save_path)
    os.rename(tmp_save_path, last_save_path)
    worst_file = self.checkpoint_files[-1][1
        ] if self.checkpoint_files else None
    if len(self.checkpoint_files
        ) < self.max_history or metric is None or self.cmp(metric, worst_file):
        if len(self.checkpoint_files) >= self.max_history:
            self._cleanup_checkpoints(1)
        filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
        save_path = os.path.join(self.checkpoint_dir, filename)
        os.link(last_save_path, save_path)
        self.checkpoint_files.append((save_path, metric))
        self.checkpoint_files = sorted(self.checkpoint_files, key=lambda x:
            x[1], reverse=not self.decreasing)
        checkpoints_str = 'Current checkpoints:\n'
        for c in self.checkpoint_files:
            checkpoints_str += ' {}\n'.format(c)
        LOGGER.info(checkpoints_str)
        if metric is not None and (self.best_metric is None or self.cmp(
            metric, self.best_metric)):
            self.best_epoch = epoch
            self.best_metric = metric
            best_save_path = os.path.join(self.checkpoint_dir, 'model_best' +
                self.extension)
            if os.path.exists(best_save_path):
                os.unlink(best_save_path)
            os.link(last_save_path, best_save_path)
    return (None, None) if self.best_metric is None else (self.best_metric,
        self.best_epoch)
