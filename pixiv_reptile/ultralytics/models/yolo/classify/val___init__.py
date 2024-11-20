def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None,
    _callbacks=None):
    """Initializes ClassificationValidator instance with args, dataloader, save_dir, and progress bar."""
    super().__init__(dataloader, save_dir, pbar, args, _callbacks)
    self.targets = None
    self.pred = None
    self.args.task = 'classify'
    self.metrics = ClassifyMetrics()
