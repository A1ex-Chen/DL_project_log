def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a OBBTrainer object with given arguments."""
    if overrides is None:
        overrides = {}
    overrides['task'] = 'obb'
    super().__init__(cfg, overrides, _callbacks)
