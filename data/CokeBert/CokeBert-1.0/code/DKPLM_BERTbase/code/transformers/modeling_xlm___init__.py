def __init__(self, config):
    super(XLMForQuestionAnswering, self).__init__(config)
    self.transformer = XLMModel(config)
    self.qa_outputs = SQuADHead(config)
    self.init_weights()
