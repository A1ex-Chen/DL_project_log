def save_recovery(self, epoch, batch_idx=0):
    assert epoch >= 0
    filename = '-'.join([self.recovery_prefix, str(epoch), str(batch_idx)]
        ) + self.extension
    save_path = os.path.join(self.recovery_dir, filename)
    self._save(save_path, epoch)
    if os.path.exists(self.last_recovery_file):
        try:
            LOGGER.debug('Cleaning recovery: %s', self.last_recovery_file)
            os.remove(self.last_recovery_file)
        except Exception as e:
            LOGGER.error("Exception '%s' while removing %s", e, self.
                last_recovery_file)
    self.last_recovery_file = self.curr_recovery_file
    self.curr_recovery_file = save_path
