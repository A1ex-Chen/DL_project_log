def __init__(self, config):
    super(BertForQuestionAnswering, self).__init__(config)
    self.num_labels = config.num_labels
    self.bert = BertModel(config)
    self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
    self.init_weights()
