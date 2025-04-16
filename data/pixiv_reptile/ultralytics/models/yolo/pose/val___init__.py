def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None,
    _callbacks=None):
    """Initialize a 'PoseValidator' object with custom parameters and assigned attributes."""
    super().__init__(dataloader, save_dir, pbar, args, _callbacks)
    self.sigma = None
    self.kpt_shape = None
    self.args.task = 'pose'
    self.metrics = PoseMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
    if isinstance(self.args.device, str) and self.args.device.lower() == 'mps':
        LOGGER.warning(
            "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. See https://github.com/ultralytics/ultralytics/issues/4031."
            )
