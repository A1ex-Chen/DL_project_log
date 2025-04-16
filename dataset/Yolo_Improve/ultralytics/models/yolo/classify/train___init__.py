def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a ClassificationTrainer object with optional configuration overrides and callbacks."""
    if overrides is None:
        overrides = {}
    overrides['task'] = 'classify'
    if overrides.get('imgsz') is None:
        overrides['imgsz'] = 224
    super().__init__(cfg, overrides, _callbacks)
