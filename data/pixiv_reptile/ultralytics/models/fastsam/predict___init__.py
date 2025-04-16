def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """
        Initializes the FastSAMPredictor class, inheriting from DetectionPredictor and setting the task to 'segment'.

        Args:
            cfg (dict): Configuration parameters for prediction.
            overrides (dict, optional): Optional parameter overrides for custom behavior.
            _callbacks (dict, optional): Optional list of callback functions to be invoked during prediction.
        """
    super().__init__(cfg, overrides, _callbacks)
    self.args.task = 'segment'
