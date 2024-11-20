def __init__(self, config: T5Config):
    super().__init__(config)
    self.shared = nn.Embedding(config.vocab_size, config.d_model)
    encoder_config = copy.deepcopy(config)
    encoder_config.use_cache = False
    encoder_config.is_encoder_decoder = False
    self.encoder = T5Stack(encoder_config, self.shared)
    self.init_weights()
