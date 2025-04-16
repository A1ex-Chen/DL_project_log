def __init__(self, config):
    super().__init__(config)
    self.start_n_top = config.start_n_top
    self.end_n_top = config.end_n_top
    self.transformer = XLNetModel(config)
    self.start_logits = PoolerStartLogits(config)
    self.end_logits = PoolerEndLogits(config)
    self.answer_class = PoolerAnswerClass(config)
    self.init_weights()
