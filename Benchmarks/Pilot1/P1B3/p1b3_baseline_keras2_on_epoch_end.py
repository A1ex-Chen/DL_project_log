def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    epoch_log = 'Epoch {}/{}'.format(epoch + 1, self.epochs)
    for k in self.params['metrics']:
        if k in logs:
            self.log_values.append((k, logs[k]))
            epoch_log += ' - {}: {:.4f}'.format(k, logs[k])
    for k, v in self.extra_log_values:
        self.log_values.append((k, v))
        epoch_log += ' - {}: {:.4f}'.format(k, float(v))
    if self.verbose:
        self.progbar.update(self.seen, self.log_values)
    benchmark.logger.debug(epoch_log)
