def report_final(self):
    self.info('checkpoints kept: %i' % len(self.epochs))
    self.info('checkpoints list: %s' % str(self.epochs))
