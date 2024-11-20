def __init__(self, config):
    super().__init__(config)
    num_labels = getattr(config, 'num_labels', 2)
    self.num_labels = num_labels
    self.deberta = DebertaModel(config)
    self.pooler = ContextPooler(config)
    output_dim = self.pooler.output_dim
    self.classifier = torch.nn.Linear(output_dim, num_labels)
    drop_out = getattr(config, 'cls_dropout', None)
    drop_out = (self.config.hidden_dropout_prob if drop_out is None else
        drop_out)
    self.dropout = StableDropout(drop_out)
    self.init_weights()
