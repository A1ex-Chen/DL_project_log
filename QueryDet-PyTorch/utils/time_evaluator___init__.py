def __init__(self, distributed, unit, out_file=None):
    self.distributed = distributed
    self.all_time = []
    self.unit = unit
    self.out_file = out_file
    if unit not in {'minisecond', 'second'}:
        raise NotImplementedError('Unsupported time unit %s' % unit)
    self.reset()
