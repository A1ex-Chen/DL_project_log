def __init__(self, config: DPRConfig):
    super().__init__(config)
    self.config = config
    self.span_predictor = DPRSpanPredictor(config)
    self.init_weights()
