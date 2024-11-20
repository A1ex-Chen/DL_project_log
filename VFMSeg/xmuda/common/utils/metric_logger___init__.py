def __init__(self, delimiter='\t'):
    self.meters = defaultdict(AverageMeter)
    self.delimiter = delimiter
