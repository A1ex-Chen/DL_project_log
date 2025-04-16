def __init__(self, config: DPRConfig, *args, **kwargs):
    super().__init__(config, *args, **kwargs)
    self.config = config
    self.span_predictor = TFDPRSpanPredictor(config, name='span_predictor')
