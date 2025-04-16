def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initializes OBBPredictor with optional model and data configuration overrides."""
    super().__init__(cfg, overrides, _callbacks)
    self.args.task = 'obb'
