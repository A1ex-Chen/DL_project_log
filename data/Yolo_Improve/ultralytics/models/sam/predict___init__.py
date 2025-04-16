def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """
        Initialize the Predictor with configuration, overrides, and callbacks.

        The method sets up the Predictor object and applies any configuration overrides or callbacks provided. It
        initializes task-specific settings for SAM, such as retina_masks being set to True for optimal results.

        Args:
            cfg (dict): Configuration dictionary.
            overrides (dict, optional): Dictionary of values to override default configuration.
            _callbacks (dict, optional): Dictionary of callback functions to customize behavior.
        """
    if overrides is None:
        overrides = {}
    overrides.update(dict(task='segment', mode='predict', imgsz=1024))
    super().__init__(cfg, overrides, _callbacks)
    self.args.retina_masks = True
    self.im = None
    self.features = None
    self.prompts = {}
    self.segment_all = False
