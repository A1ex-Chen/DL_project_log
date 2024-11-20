def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels
    self.config = config
    self.cokebert = CokeBertModel(config)
    classifier_dropout = (config.classifier_dropout if config.
        classifier_dropout is not None else config.hidden_dropout_prob)
    self.dropout = nn.Dropout(classifier_dropout)
    self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
    self.activation = nn.Tanh()
    self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)
    self.post_init()
