def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
    super().__init__(cfg, overrides, _callbacks)
    self.args.task = 'segment'
