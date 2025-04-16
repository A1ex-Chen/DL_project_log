def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None,
    _callbacks=None):
    """Initialize OBBValidator and set task to 'obb', metrics to OBBMetrics."""
    super().__init__(dataloader, save_dir, pbar, args, _callbacks)
    self.args.task = 'obb'
    self.metrics = OBBMetrics(save_dir=self.save_dir, plot=True, on_plot=
        self.on_plot)
