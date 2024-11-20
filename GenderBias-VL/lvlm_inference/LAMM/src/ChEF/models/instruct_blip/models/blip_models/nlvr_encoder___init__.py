def __init__(self, config, add_pooling_layer=True):
    super().__init__(config)
    self.config = config
    self.embeddings = BertEmbeddings(config)
    self.encoder = BertEncoder(config)
    self.pooler = BertPooler(config) if add_pooling_layer else None
    self.init_weights()
