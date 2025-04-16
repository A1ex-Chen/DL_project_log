def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initializes PosePredictor, sets task to 'pose' and logs a warning for using 'mps' as device."""
    super().__init__(cfg, overrides, _callbacks)
    self.args.task = 'pose'
    if isinstance(self.args.device, str) and self.args.device.lower() == 'mps':
        LOGGER.warning(
            "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. See https://github.com/ultralytics/ultralytics/issues/4031."
            )
