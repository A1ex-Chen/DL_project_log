def __init__(self, log_dir=None) ->None:
    self.log_dir = log_dir
    if SummaryWriter is not None and is_primary():
        self.writer = SummaryWriter(self.log_dir)
    else:
        self.writer = None
