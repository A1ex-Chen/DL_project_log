def __init__(self, config):
    super(DistilBertForTokenClassification, self).__init__(config)
    self.num_labels = config.num_labels
    self.distilbert = DistilBertModel(config)
    self.dropout = nn.Dropout(config.dropout)
    self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    self.init_weights()
