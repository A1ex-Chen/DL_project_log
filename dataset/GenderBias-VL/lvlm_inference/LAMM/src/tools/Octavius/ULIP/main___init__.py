def __init__(self, num_batches, meters, prefix=''):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix
