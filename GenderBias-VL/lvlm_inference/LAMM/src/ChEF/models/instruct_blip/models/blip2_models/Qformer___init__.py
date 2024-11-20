def __init__(self, config):
    super().__init__(config)
    self.bert = BertModel(config, add_pooling_layer=False)
    self.cls = BertOnlyMLMHead(config)
    self.init_weights()
