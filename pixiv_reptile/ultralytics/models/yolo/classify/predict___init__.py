def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initializes ClassificationPredictor setting the task to 'classify'."""
    super().__init__(cfg, overrides, _callbacks)
    self.args.task = 'classify'
    self._legacy_transform_name = 'ultralytics.yolo.data.augment.ToTensor'
