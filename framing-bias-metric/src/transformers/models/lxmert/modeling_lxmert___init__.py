def __init__(self, config):
    super().__init__(config)
    self.config = config
    self.num_qa_labels = config.num_qa_labels
    self.visual_loss_normalizer = config.visual_loss_normalizer
    self.lxmert = LxmertModel(config)
    self.answer_head = LxmertVisualAnswerHead(config, self.num_qa_labels)
    self.init_weights()
    self.loss = CrossEntropyLoss()
