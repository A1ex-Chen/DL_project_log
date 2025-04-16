def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a PoseTrainer object with specified configurations and overrides."""
    if overrides is None:
        overrides = {}
    overrides['task'] = 'pose'
    super().__init__(cfg, overrides, _callbacks)
    if isinstance(self.args.device, str) and self.args.device.lower() == 'mps':
        LOGGER.warning(
            "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. See https://github.com/ultralytics/ultralytics/issues/4031."
            )
