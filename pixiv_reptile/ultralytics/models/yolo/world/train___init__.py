def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """Initialize a WorldTrainer object with given arguments."""
    if overrides is None:
        overrides = {}
    super().__init__(cfg, overrides, _callbacks)
    try:
        import clip
    except ImportError:
        checks.check_requirements('git+https://github.com/ultralytics/CLIP.git'
            )
        import clip
    self.clip = clip
