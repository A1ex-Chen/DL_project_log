def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
        overrides = {}
    super().__init__(cfg, overrides, _callbacks)
