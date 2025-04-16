def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels
    self.roberta = RobertaModel(config, add_pooling_layer=False)
    self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
    self.init_weights()
