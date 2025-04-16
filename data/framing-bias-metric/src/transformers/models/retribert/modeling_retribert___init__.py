def __init__(self, config):
    super().__init__(config)
    self.projection_dim = config.projection_dim
    self.bert_query = BertModel(config)
    self.bert_doc = None if config.share_encoders else BertModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.project_query = nn.Linear(config.hidden_size, config.
        projection_dim, bias=False)
    self.project_doc = nn.Linear(config.hidden_size, config.projection_dim,
        bias=False)
    self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
    self.init_weights()
