def keep(self, epoch, epoch_now, kept):
    """
        kept: Number of epochs already kept
        return True if we are keeping this epoch, else False
        """
    if epoch == epoch_now:
        self.debug('keep(): epoch is latest: %i' % epoch)
        return True
    if self.epoch_best == epoch:
        self.debug('keep(): epoch is best: %i' % epoch)
        return True
    if kept < self.keep_limit:
        self.debug('keep(): epoch count is < limit %i' % self.keep_limit)
        return True
    return False
