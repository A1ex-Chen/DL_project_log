def clean(self, epoch_now):
    """
        Clean old epoch directories
              in accordance with ckpt_keep policies.
        Return number of checkpoints kept and deleted
        """
    deleted = 0
    kept = 0
    for epoch in reversed(self.epochs):
        self.debug('clean(): checking epoch directory: %i' % epoch)
        if not self.keep(epoch, epoch_now, kept):
            deleted += 1
            self.delete(epoch)
            self.debug('clean(): deleted epoch: %i' % epoch)
        else:
            kept += 1
    return kept, deleted
