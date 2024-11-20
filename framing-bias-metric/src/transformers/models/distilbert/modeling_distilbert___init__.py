def __init__(self, config):
    super().__init__(config)
    self.distilbert = DistilBertModel(config)
    self.pre_classifier = nn.Linear(config.dim, config.dim)
    self.classifier = nn.Linear(config.dim, 1)
    self.dropout = nn.Dropout(config.seq_classif_dropout)
    self.init_weights()
