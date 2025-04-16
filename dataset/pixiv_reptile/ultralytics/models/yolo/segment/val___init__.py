def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None,
    _callbacks=None):
    """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics."""
    super().__init__(dataloader, save_dir, pbar, args, _callbacks)
    self.plot_masks = None
    self.process = None
    self.args.task = 'segment'
    self.metrics = SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
