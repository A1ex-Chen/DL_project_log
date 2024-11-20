def save_check(self, logs, epoch):
    """
        Make sure we want to save this epoch based on the
        model metrics in given logs
        Also updates epoch_best if appropriate
        """
    if self.save_interval == 0:
        return False
    if epoch < self.skip_epochs:
        self.debug('model saving disabled until epoch %d' % self.skip_epochs)
        return False
    if self.save_check_best(logs, epoch):
        self.epoch_best = epoch
        return True
    if epoch == self.epoch_max:
        self.info('writing final epoch %i ...' % epoch)
        return True
    if epoch % self.save_interval == 0:
        return True
    self.debug('not writing this epoch.')
    return False
